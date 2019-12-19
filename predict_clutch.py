import pickle
import glob
import time

import numpy as np
import imageio
import matplotlib.pyplot as plt

if __name__ == "__main__":

    with open("./data/clutch.pkl", "rb") as f:
        clutch_root_poses, clutch_hand_angles, clutch_joint_angles = pickle.load(f)

    mean_clutch_joint_angles = np.mean(clutch_joint_angles, axis=0)

    data_filenames = sorted(glob.glob("./data/sequence/scissors_paper/00/angles/*.pkl"))
    frame_filenames = sorted(glob.glob("./data/sequence/scissors_paper/00/frames/*.jpg"))
    # data_filenames = sorted(glob.glob("./data/dynamic/scissors/04/angles/*.pkl"))
    # data_filenames = sorted(glob.glob("./data/dynamic/scissors/03/angles/*.pkl"))
    print(data_filenames)
    plt.figure()
    # plt.show()
    for i, (data_filename, frame_filename) in enumerate(zip(data_filenames, frame_filenames)):
        with open(data_filename, "rb") as f:
            data_root_poses, data_hand_angles, data_joint_angles = pickle.load(f)

            # L2-distance
            # distance_sq = np.sum((data_joint_angles - mean_clutch_joint_angles) ** 2)
            # L1-distance
            distance = np.sum(np.abs(data_joint_angles - mean_clutch_joint_angles))
            # cosine-distance
            # distance = (data_joint_angles @ mean_clutch_joint_angles) / \
            #            (np.linalg.norm(data_joint_angles) * np.linalg.norm(mean_clutch_joint_angles))

            img = imageio.imread(frame_filename)
            plt.imshow(img)
            plt.draw()
            print(i, distance, distance < 2)

            plt.pause(0.1)