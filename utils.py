import glob
import os
import pickle
import zlib

import cv2
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation


def get_angle(p0, p1, p2):
    """
    Compute the angle between p0 and p2, about p1's reference frame.
    Assume that p0 and p1 are connected, and p1 and p2 are connected.
    """
    u = p0 - p1
    u /= np.linalg.norm(u)
    v = p2 - p1
    v /= np.linalg.norm(v)

    return np.arccos(u @ v)


def get_orthogonal_vector(v):
    x = abs(v[0])
    y = abs(v[1])
    z = abs(v[2])
    if x < y:
        if x < z:
            other = [1, 0, 0]
        else:
            other = [0, 0, 1]
    else:
        if y < z:
            other = [0, 1, 0]
        else:
            other = [0, 0, 1]
    other = np.array(other)
    return np.cross(v, other)


def get_quaternion(p0, p1, p2):
    v0 = p0 - p1
    v1 = p2 - p1
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    if np.all(v0 == -v1):
        xyz = get_orthogonal_vector(v0)
        w = [0]
        basis = np.concatenate([xyz, w])
        return Rotation.from_quat(basis).as_quat()

    half = v0 + v1
    half = half / np.linalg.norm(half)

    xyz = np.cross(v0, half)
    w = [np.dot(v0, half)]
    basis = np.concatenate([xyz, w])
    return Rotation.from_quat(basis).as_quat()


# define all bone connections
BONES = [(0, 4),
         (4, 3),
         (3, 2),
         (2, 1),

         (0, 8),
         (8, 7),
         (7, 6),
         (6, 5),

         (0, 12),
         (12, 11),
         (11, 10),
         (10, 9),

         (0, 16),
         (16, 15),
         (15, 14),
         (14, 13),

         (0, 20),
         (20, 19),
         (19, 18),
         (18, 17)]
BONES = np.asarray(BONES)

# define all triplets
TRIPLETS = [
    (0, 4, 3), (4, 3, 2), (3, 2, 1),
    (0, 8, 7), (8, 7, 6), (7, 6, 5),
    (0, 12, 11), (12, 11, 10), (11, 10, 9),
    (0, 16, 15), (16, 15, 14), (15, 14, 13),
    (0, 20, 19), (20, 19, 18), (19, 18, 17)
]
TRIPLETS = np.asarray(TRIPLETS)

ZIMMERMAN_TO_MANO_DICT = dict({
    0: 0,
    8: 1, 7: 2, 6: 3,
    12: 4, 11: 5, 10: 6,
    16: 10, 15: 11, 14: 12,
    20: 7, 19: 8, 18: 9,
    4: 13, 3: 14, 2: 15
})
# ZIMMERMAN_TO_MANO_DICT = dict({
#         0: 0,
#         8: 1, 7: 2, 6: 3,
#         12: 4, 11: 5, 10: 6,
#         16: 7, 15: 8, 14: 9,
#         20: 10, 19: 11, 18: 12,
#         4: 13, 3: 14, 2: 15
#     })
MANO_TO_ZIMMERMAN_DICT = dict({v: k for k, v in ZIMMERMAN_TO_MANO_DICT.items()})
MANO_TO_ZIMMERMAN_INDICES = np.stack([v for v in MANO_TO_ZIMMERMAN_DICT.values()])

EXTREMETY_IDX = np.array([5, 8, 11, 14])
CLUTCH_DIST_THRES = 5
CLUTCH_STD_THRES = 1


def visualize_angles(plt, plt_name, joint_angles_seq, hand_angles_seq):
    plt.figure()
    plt.title("Angle Sequence for {:s}".format(plt_name))
    plt.xlabel("Frame")
    plt.ylabel("Angle (rad)")
    # plt.ylim(-np.pi, np.pi)
    for i in range(joint_angles_seq.shape[1]):
        plt.plot(joint_angles_seq[:, i])
    #     plt.plot(velocities_seq[:, i])
    for i in range(hand_angles_seq.shape[1]):
        plt.plot(hand_angles_seq[:, i])

    # for i in range(3):
    #     plt.plot(root_pose_seq[:, i])
    # print(root_pose_seq)

    plt.show()


def convert_coords_to_angles(coords, scale, center):
    root_pose = np.concatenate([center.reshape(-1), scale.reshape(-1)])
    # figure out pitch, yaw, and roll of hand root
    basis = get_rotation_basis(coords)
    x_angle, y_angle, z_angle = Rotation.from_dcm(basis.T).as_euler('XYZ')
    hand_angles = np.array([x_angle, y_angle, z_angle])
    triplets_of_coords = coords[TRIPLETS]
    # print(triplets_of_coords)
    joint_angles = np.zeros(TRIPLETS.shape[0])
    for i, triplet_of_coords in enumerate(triplets_of_coords):
        p0, p1, p2 = triplet_of_coords
        angle = get_angle(p0, p1, p2)
        joint_angles[i] = angle
    return root_pose, hand_angles, joint_angles


def convert_coords_to_quaternions(coords, scale, center):
    root_pose = np.concatenate([center.reshape(-1), scale.reshape(-1)])
    # figure out pitch, yaw, and roll of hand root
    basis = get_rotation_basis(coords)
    hand_quat = Rotation.from_dcm(basis.T).as_quat()
    triplets_of_coords = coords[TRIPLETS]
    # print(triplets_of_coords)
    joint_quats = np.zeros([TRIPLETS.shape[0], 4])
    for i, triplet_of_coords in enumerate(triplets_of_coords):
        p0, p1, p2 = triplet_of_coords
        quat = get_quaternion(p0, p1, p2)
        joint_quats[i] = quat
    return root_pose, hand_quat, joint_quats


# def convert_quaternions_to_coords(template_coords, hand_quat, joint_quats):
#     coords = np.copy(template_coords)
#
#     # translate hand to origin
#     coords = coords - coords[0]
#
#     # Undo any previous rotation to input hand
#     basis = get_rotation_basis(coords)
#     coords = coords @ basis
#
#     # Rotate hand to new root orientation
#     basis = Rotation.from_quat(hand_quat).as_dcm()
#     coords = coords @ basis
#
#     # Rotate fingers into position
#     assert(len(joint_quats) == len(TRIPLETS))
#     # NOTE: this assumes that the joints in TRIPLETS are ordered
#     # from the root to the fingertip, for each finger.
#     # In other words, it is assumed that no bone will appear before
#     # any other bone that is closer to the root and to which it is connected
#     for i in range(len(joint_quats)):
#         j0, j1, j2 = TRIPLETS[i]
#         prev_bone_vector = coords[j0] - coords[j1]
#         next_bone_vector = coords[j2] - coords[j1]
#         bone_rotation_matrix = Rotation.from_quat(joint_quats[i]).as_dcm()
#         new_next_bone_vector = prev_bone_vector @ bone_rotation_matrix
#         new_next_bone_length = np.linalg.norm(new_next_bone_vector)
#         next_bone_length = np.linalg.norm(next_bone_vector)
#         new_next_bone_vector = new_next_bone_vector / new_next_bone_length * next_bone_length
#         coords[j2] = coords[j1] + new_next_bone_vector
#
#     return coords

def convert_coords_to_rotationally_invariant_displacements(coords):
    basis = get_rotation_basis(coords)
    rotated_coords = coords @ basis
    displacements = []
    for bone in BONES:
        p1, p2 = rotated_coords[np.asarray(bone)]
        displacements.append(p2 - p1)
    displacements = np.concatenate(displacements)
    return displacements


def convert_rotationally_invariant_displacements_to_coords(displacements):
    assert (len(displacements) == len(BONES) * 3)
    displacements = displacements.reshape(-1, 3)
    coords = np.zeros((len(BONES) + 1, 3))
    for i in range(len(BONES)):
        b0, b1 = BONES[i]
        coords[b1] = coords[b0] + displacements[i]
    coords[:, 1] *= -1
    return coords


def get_rotation_basis(coords):
    triangle = coords[np.array([0, 8, 20])]
    p0, p1, p2 = triangle
    u = p0 - p1
    v = p2 - p1
    x_axis = u
    y_axis = np.cross(u, v)
    z_axis = np.cross(x_axis, y_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)
    basis = np.stack([x_axis, y_axis, z_axis]).T
    return basis


def get_clutch_distance(current_features, clutch_features, weights, metric="l1"):
    weights /= np.linalg.norm(weights)
    mean_clutch_features = np.mean(clutch_features, axis=0)
    residuals = mean_clutch_features - current_features
    if metric == "l1":
        dist = np.abs(residuals) @ weights
    elif metric == "l2":
        dist = (residuals ** 2) @ weights
    elif metric == "l1/2":
        dist = (np.abs(residuals) ** (1. / 2)) @ weights
    else:
        dist = np.Inf
    return np.min(dist)


def detect_clutch(clutch_distance_buffer):
    avg_distance = np.mean(clutch_distance_buffer)
    std = np.std(clutch_distance_buffer)
    dist_good_yes = avg_distance < CLUTCH_DIST_THRES
    buffer_full_yes = len(clutch_distance_buffer) == clutch_distance_buffer.maxlen
    std_good_yes = std < CLUTCH_STD_THRES
    return dist_good_yes and buffer_full_yes and std_good_yes


def predict_gesture(gesture_classifier, gesture_features_buffer):
    gesture_features = np.stack(gesture_features_buffer)
    return gesture_classifier.predict_gesture(gesture_features)


def compress_and_send_to_socket(data, socket):
    c2s_data = b""
    c2s_data = cv2.imencode(".jpg", data)[1].tostring()
    c2s_data_compressed = zlib.compress(c2s_data)
    c2s_data_to_send = b"<START>" + c2s_data_compressed + b"<END>"
    # tic = time.time()
    socket.sendall(c2s_data_to_send)


def receive_from_socket(s, initial_data=b""):
    s2c_data = initial_data
    # print("Client: start blocking")
    while b"<END>" not in s2c_data:
        received = s.recv(1024)
        s2c_data += received
        if not received:
            s2c_data = b""
            # print("Client: received end of transmission")
            # s.close()
            continue
        # print("Client: receive chunk")
    s2c_data_trimmed = s2c_data.replace(b"<START>", b"")
    s2c_data_trimmed = s2c_data_trimmed.split(b"<END>")[0]
    s2c_data_remaining = b"".join(s2c_data.split(b"<END>")[1:])
    return s2c_data_trimmed, s2c_data_remaining


def load_gesture_lists(gesture_label, indices):
    root_pose_list = []
    joint_angles_list = []
    hand_angles_list = []
    velocities_list = []
    for index in indices:
        angles_filenames = sorted(glob.glob("./data/dynamic/{:s}/{:02d}/angles/*.pkl".format(gesture_label, index)))
        index_joint_angles_list = []
        index_hand_angles_list = []
        index_root_pose_list = []
        for angles_filename in angles_filenames:
            with open(angles_filename, 'rb') as f:
                root_pose, hand_angles, joint_angles = pickle.load(f)
            index_joint_angles_list.append(joint_angles)
            index_hand_angles_list.append(hand_angles)
            index_root_pose_list.append(root_pose)

        index_joint_angles = np.stack(index_joint_angles_list)
        index_hand_angles = np.stack(index_hand_angles_list)
        index_root_pose = np.stack(index_root_pose_list)
        index_velocities_list = []
        for angles_before, angles_after in zip(index_joint_angles, index_joint_angles[1:]):
            velocities = angles_after - angles_before
            index_velocities_list.append(velocities)
        index_velocities = np.stack(index_velocities_list)

        joint_angles_list.append(index_joint_angles)
        hand_angles_list.append(index_hand_angles)
        root_pose_list.append(index_root_pose)
        velocities_list.append(index_velocities)

    return joint_angles_list, velocities_list, hand_angles_list, root_pose_list


def get_clutch_data():
    clutch_filenames = sorted(glob.glob("./data/dynamic/clutch/**/angles/*.pkl"))
    root_poses_seq = []
    hand_angles_seq = []
    joint_angles_seq = []
    for clutch_filename in clutch_filenames:
        with open(clutch_filename, "rb") as f:
            root_pose, hand_angles, joint_angles = pickle.load(f)
        root_poses_seq.append(root_pose)
        hand_angles_seq.append(hand_angles)
        joint_angles_seq.append(joint_angles)
    root_poses_seq = np.stack(root_poses_seq)
    hand_angles_seq = np.stack(hand_angles_seq)
    joint_angles_seq = np.stack(joint_angles_seq)
    return hand_angles_seq, joint_angles_seq, root_poses_seq


def get_all_dynamic_data():
    filenames = sorted(glob.glob("./data/dynamic/**/**/angles/*.pkl"))
    root_poses_seq = []
    hand_angles_seq = []
    joint_angles_seq = []
    for filename in filenames:
        with open(filename, "rb") as f:
            root_pose, hand_angles, joint_angles = pickle.load(f)
        root_poses_seq.append(root_pose)
        hand_angles_seq.append(hand_angles)
        joint_angles_seq.append(joint_angles)
    root_poses_seq = np.stack(root_poses_seq)
    hand_angles_seq = np.stack(hand_angles_seq)
    joint_angles_seq = np.stack(joint_angles_seq)
    return hand_angles_seq, joint_angles_seq, root_poses_seq


def get_reference_joint_angles():
    ref_pose_vid_dir = "./data/dynamic/clutch/99/angles/*.pkl"
    ref_filenames = sorted(glob.glob(ref_pose_vid_dir))
    root_poses_seq = []
    hand_angles_seq = []
    joint_angles_seq = []
    for ref_filename in ref_filenames:
        with open(ref_filename, "rb") as f:
            root_pose, hand_angles, joint_angles = pickle.load(f)
        root_poses_seq.append(root_pose)
        hand_angles_seq.append(hand_angles)
        joint_angles_seq.append(joint_angles)
    joint_angles_seq = np.stack(joint_angles_seq)
    # get the reference
    ref_joint_angles = np.mean(joint_angles_seq[:10, :], axis=0)
    # ref_joint_angles = np.mean(joint_angles_seq, axis=0)
    return ref_joint_angles


def get_reference_joint_coords():
    ref_pose_vid_dir = "./data/dynamic/clutch/99/coords/*.pkl"
    ref_filenames = sorted(glob.glob(ref_pose_vid_dir))
    coord_seq = []
    for ref_filename in ref_filenames:
        with open(ref_filename, "rb") as f:
            scale, center, keypoints_3d = pickle.load(f)
        coord_seq.append(keypoints_3d)
    coord_seq = np.stack(coord_seq)
    # get the reference
    ref_joint_coords = np.mean(coord_seq[:10], axis=0)
    print(ref_joint_coords.shape)
    return ref_joint_coords


def load_angles(gesture_name, index):
    joint_angles_seq = []
    hand_angles_seq = []
    root_poses_seq = []
    angles_filenames = sorted(glob.glob("./data/dynamic/{:s}/{:02d}/angles/*.pkl".format(gesture_name, index)))
    if len(angles_filenames) == 0:
        raise RuntimeError("gesture_name={:s}, index={:02d}".format(gesture_name, index))
    for angles_filename in angles_filenames:
        with open(angles_filename, 'rb') as f:
            root_pose, hand_angles, joint_angles = pickle.load(f)
        joint_angles_seq.append(joint_angles)
        hand_angles_seq.append(hand_angles)
        root_poses_seq.append(root_pose)
    joint_angles_seq = np.stack(joint_angles_seq)
    hand_angles_seq = np.stack(hand_angles_seq)
    root_poses_seq = np.stack(root_poses_seq)
    return joint_angles_seq, hand_angles_seq, root_poses_seq


def load_coords(gesture_name, index):
    scales = []
    centers = []
    coords = []
    coord_filenames = sorted(glob.glob("./data/dynamic/{:s}/{:02d}/coords/*.pkl".format(gesture_name, index)))
    if len(coord_filenames) == 0:
        raise RuntimeError("gesture_name={:s}, index={:02d}".format(gesture_name, index))
    for coord_filename in coord_filenames:
        with open(coord_filename, 'rb') as f:
            scale_v, center_v, keypoint_coord3d_v = pickle.load(f)
        scales.append(scale_v)
        centers.append(center_v)
        coords.append(keypoint_coord3d_v)
    scales = np.stack(scales)
    centers = np.stack(centers)
    coords = np.stack(coords)
    return scales, centers, coords


def load_coords(gesture_name, index):
    scales = []
    centers = []
    coords = []
    coord_filenames = sorted(glob.glob("./data/dynamic/{:s}/{:02d}/coords/*.pkl".format(gesture_name, index)))
    if len(coord_filenames) == 0:
        raise RuntimeError("gesture_name={:s}, index={:02d}".format(gesture_name, index))
    for coord_filename in coord_filenames:
        with open(coord_filename, 'rb') as f:
            scale_v, center_v, keypoint_coord3d_v = pickle.load(f)
        scales.append(scale_v)
        centers.append(center_v)
        coords.append(keypoint_coord3d_v)
    scales = np.stack(scales)
    centers = np.stack(centers)
    coords = np.stack(coords)
    return scales, centers, coords


def load_quats(gesture_name, index):
    joint_quats_seq = []
    hand_quats_seq = []
    root_poses_seq = []
    quats_filenames = sorted(glob.glob("./data/dynamic/{:s}/{:02d}/quats/*.pkl".format(gesture_name, index)))
    if len(quats_filenames) == 0:
        raise RuntimeError("gesture_name={:s}, index={:02d}".format(gesture_name, index))
    for quats_filename in quats_filenames:
        with open(quats_filename, 'rb') as f:
            root_pose, hand_quats, joint_quats = pickle.load(f)
        joint_quats_seq.append(joint_quats)
        hand_quats_seq.append(hand_quats)
        root_poses_seq.append(root_pose)
    joint_quats_seq = np.stack(joint_quats_seq)
    hand_quats_seq = np.stack(hand_quats_seq)
    root_poses_seq = np.stack(root_poses_seq)
    return joint_quats_seq, hand_quats_seq, root_poses_seq


def load_disps(gesture_name, index):
    disps_seq = []
    disps_filenames = sorted(glob.glob("./data/dynamic/{:s}/{:02d}/disps/*.pkl".format(gesture_name, index)))
    if len(disps_filenames) == 0:
        raise RuntimeError("gesture_name={:s}, index={:02d}".format(gesture_name, index))
    for disps_filename in disps_filenames:
        with open(disps_filename, 'rb') as f:
            disps = pickle.load(f)
        disps_seq.append(disps)
    disps_seq = np.stack(disps_seq)
    return disps_seq


def load_all_coords(gesture_name):
    n_examples = len(os.listdir("./data/dynamic/{:s}/".format(gesture_name)))
    all_scales = []
    all_centers = []
    all_coords = []
    for i in range(n_examples):
        scales, centers, coords = load_coords(gesture_name, i)
        all_scales.append(scales)
        all_centers.append(centers)
        all_coords.append(coords)
    return all_scales, all_centers, all_coords


def load_all_quats(gesture_name):
    n_examples = len(os.listdir("./data/dynamic/{:s}/".format(gesture_name)))
    all_joint_quats = []
    all_hand_quats = []
    all_root_poses = []
    for i in range(n_examples):
        joint_quats, hand_quats, root_poses = load_quats(gesture_name, i)
        all_joint_quats.append(joint_quats)
        all_hand_quats.append(hand_quats)
        all_root_poses.append(root_poses)
    return all_joint_quats, all_hand_quats, all_root_poses


def load_all_disps(gesture_name):
    n_examples = len(os.listdir("./data/dynamic/{:s}/".format(gesture_name)))
    all_disps = []
    for i in range(n_examples):
        disps = load_disps(gesture_name, i)
        all_disps.append(disps)
    return all_disps
