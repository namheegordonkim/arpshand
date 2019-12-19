import glob
import pickle
from collections import deque

import imageio
import matplotlib.pyplot as plt
import numpy as np

from utils import get_clutch_distance, detect_clutch, predict_gesture

CLUTCH_THRES = 1.0

if __name__ == "__main__":

    with open("./data/clutch.pkl", "rb") as f:
        clutch_root_poses, clutch_hand_angles, clutch_joint_angles = pickle.load(f)

    with open("./models/onedollar.pkl", "rb") as f:
        gesture_classifier = pickle.load(f)

    mean_clutch_joint_angles = np.mean(clutch_joint_angles, axis=0)
    data_filenames = sorted(glob.glob("./data/sequence/scissors_paper/03/angles/*.pkl"))
    frame_filenames = sorted(glob.glob("./data/sequence/scissors_paper/03/frames/*.jpg"))
    fig, ((im_ax, all_angle_ax), (current_angle_ax, gesture_angle_ax)) = plt.subplots(nrows=2, ncols=2)
    # im_ax.title.set_text("Image")
    # all_angle_ax.title.set_text("All Angles Sequence")
    # current_angle_ax.title.set_text("Angles")
    # gesture_angle_ax.title.set_text("Gesture Angles Sequence")

    # plt.show()

    # recognition machine states
    clutch_engaged = False
    clutch_distance_buffer = deque(maxlen=6)
    clutch_started = False

    all_angles_buffer = deque()
    gesture_features_buffer = deque(maxlen=256)

    for i, (data_filename, frame_filename) in enumerate(zip(data_filenames, frame_filenames)):
        with open(data_filename, "rb") as f:
            data_root_poses, data_hand_angles, data_joint_angles = pickle.load(f)
            all_angles_buffer.append(data_joint_angles)
            # L2-distance
            # distance_sq = np.sum((data_joint_angles - mean_clutch_joint_angles) ** 2)
            # L1-distance
            # clutch_distance = np.sum(np.abs(data_joint_angles - mean_clutch_joint_angles))
            clutch_distance = get_clutch_distance(data_joint_angles, clutch_joint_angles)
            # cosine-distance
            # distance = (data_joint_angles @ mean_clutch_joint_angles) / \
            #            (np.linalg.norm(data_joint_angles) * np.linalg.norm(mean_clutch_joint_angles))
            clutch_distance_buffer.append(clutch_distance)
            img = imageio.imread(frame_filename)
            clutch_yes = clutch_distance < CLUTCH_THRES
            mean_clutch_distance = np.mean(clutch_distance_buffer)
            print(i, clutch_distance, clutch_yes, mean_clutch_distance, clutch_engaged)

            # engage clutch if average distance in the past few poses is small
            if detect_clutch(clutch_distance_buffer):
                if clutch_engaged and not clutch_started:
                    clutch_engaged = False
                    clutch_distance_buffer.clear()
                    for _ in range(10):
                        if len(gesture_features_buffer) > 0:
                            gesture_features_buffer.pop()
                    pred = predict_gesture(gesture_classifier, gesture_features_buffer)
                    gesture_features_buffer.clear()
                    print(pred)
                    continue

                if not clutch_started:
                    clutch_started = True

            else:
                if clutch_started:
                    clutch_distance_buffer.clear()
                    for _ in range(int(len(gesture_features_buffer) / 2)):
                        gesture_features_buffer.pop()
                    clutch_started = False
                    clutch_engaged = True

            if clutch_engaged or clutch_started:
                gesture_features_buffer.append(data_joint_angles)

            im_ax.imshow(img)
            all_angle_ax.cla()
            all_angle_ax.set_ylim(1, 4)
            all_angles_np = np.stack(all_angles_buffer)
            _, n_joints = all_angles_np.shape
            for j in range(n_joints):
                all_angle_ax.plot(all_angles_np[:, j])
            current_angle_ax.cla()
            current_angle_ax.set_ylim(1, 4)
            current_angle_ax.bar(range(len(data_joint_angles)), data_joint_angles)

            gesture_angle_ax.cla()
            gesture_angle_ax.set_ylim(1, 4)
            if len(gesture_features_buffer) > 0:
                gesture_angles_np = np.stack(gesture_features_buffer)
                for j in range(n_joints):
                    gesture_angle_ax.plot(gesture_angles_np[:, j])

            plt.draw()
            plt.pause(0.0001)
