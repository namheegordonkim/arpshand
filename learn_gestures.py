import argparse
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from utils import visualize_angles


def load_gesture(gesture_label, indices):
    joint_angles_list, velocities_list, hand_angles_list, root_pose_list = load_gesture_lists(gesture_label, indices)

    joint_angles_seq = np.stack(joint_angles_list)
    velocities_seq = np.stack(velocities_list)
    hand_angles_seq = np.stack(hand_angles_list)
    root_pose_seq = np.stack(root_pose_list)

    return joint_angles_seq, velocities_seq, hand_angles_seq, root_pose_seq


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


def get_avg_start_end(seq, window_length):
    """
    Takes an n X d array and returns a 2*d X 1 array.
    First d elements is the average of first window_length elements
    Second d elements is the average of last window_length elements
    """
    first = np.mean(seq[:window_length], axis=0)
    second = np.mean(seq[-window_length:], axis=0)
    return np.concatenate([first, second])


def load_gesture_start_end(gesture_label, indices, window_length=5):
    """
    Return a 2-tuple of features per each datapoint: avg. start pose and avg. end pose
    Concatenate them into just one row before returning
    """
    joint_angles_list, velocities_list, hand_angles_list, root_pose_list = load_gesture_lists(gesture_label,
                                                                                              indices)
    joint_angles_start_end = [get_avg_start_end(joint_angles, window_length) for joint_angles in joint_angles_list]
    velocities_start_end = [get_avg_start_end(velocities, window_length) for velocities in velocities_list]
    hand_angles_start_end = [get_avg_start_end(hand_angles, window_length) for hand_angles in hand_angles_list]
    root_poses_start_end = [get_avg_start_end(root_pose, window_length) for root_pose in root_pose_list]

    joint_angles_start_end = np.stack(joint_angles_start_end)
    velocities_start_end = np.stack(velocities_start_end)
    hand_angles_start_end = np.stack(hand_angles_start_end)
    root_poses_start_end = np.stack(root_poses_start_end)

    return joint_angles_start_end, velocities_start_end, hand_angles_start_end, root_poses_start_end


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)

    args = parser.parse_args()

    with open("./configs/learn_gestures.yml", "r") as f:
        config_dict = yaml.load(f, yaml.SafeLoader)[args.config_name]

    labels = np.asarray(config_dict["labels"])
    training_indices = np.asarray(config_dict["training_indices"])
    testing_indieces = np.asarray(config_dict["testing_indices"])

    label_vectors = []
    feature_list = []
    for i, (label, indices) in enumerate(zip(labels, training_indices)):
        if len(indices) == 0:
            continue
        # features = load_gesture(label, indices)
        features = load_gesture_start_end(label, indices)
        label_vector = np.ones(features[0].shape[0], dtype=np.int) * i

        label_vectors.append(label_vector)
        feature_list.append(features[0])

    scaler = StandardScaler()
    model = KNeighborsClassifier(n_neighbors=1)

    X = np.concatenate(feature_list)
    y = np.concatenate(label_vectors)

    Z = scaler.fit_transform(X)
    model.fit(Z, y)

    test_label_vectors = []
    test_feature_list = []
    for i, (label, indices) in enumerate(zip(labels, testing_indieces)):
        if len(indices) == 0:
            continue
        # features = load_gesture(label, indices)
        features = load_gesture_start_end(label, indices)
        label_vector = np.ones(features[0].shape[0], dtype=np.int) * i

        test_label_vectors.append(label_vector)
        test_feature_list.append(features[0])

    y_tilde = np.concatenate(test_label_vectors)
    X_hat = np.concatenate(test_feature_list)
    Z_hat = scaler.transform(X_hat)
    y_hat = model.predict(Z_hat)
    print(y_hat)
    print(y_hat == y_tilde)
