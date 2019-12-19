import argparse
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from oneeuro import OneEuroFilter
from utils import get_clutch_distance, CLUTCH_DIST_THRES


def load_angles(gesture_name, index):
    joint_angles_seq = []
    hand_angles_seq = []
    root_poses_seq = []
    angles_filenames = sorted(glob.glob("./data/dynamic/{:s}/{:02d}/*.pkl".format(gesture_name, index)))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles_dir", required=True, type=str)

    args = parser.parse_args()
    angles_dir = args.angles_dir

    with open("./data/clutch.pkl", "rb") as f:
        _, _, clutch_corrected_joint_angles = pickle.load(f)
    with open("./data/clutch_pca_model.pkl", "rb") as f:
        clutch_model = pickle.load(f)
    with open("./data/clutch_pca_coeffs.pkl", "rb") as f:
        clutch_coeffs = pickle.load(f)
    with open("./models/clutch_corrector.pkl", "rb") as f:
        clutch_corrector_model = pickle.load(f)
    with open("./models/clutch_scaler.pkl", "rb") as f:
        clutch_scaler = pickle.load(f)


    joint_angles_seq, hand_angles_seq, root_poses_seq = load_angles()

    corrector_input_features = np.concatenate([joint_angles_seq, hand_angles_seq, root_poses_seq], axis=1)
    n, d = corrector_input_features.shape
    min_cutoff = 0.75
    beta = 0.007
    filters = [OneEuroFilter(15, min_cutoff, beta, 0.1) for _ in range(21)]
    corrector_input_features_filtered = np.zeros([n, d])
    for i in range(n):
        for j in range(d):
            corrector_input_features_filtered[i, j] = filters[j].filter(corrector_input_features[i, j])
    # input_features_filtered = input_features

    estimated_residuals = clutch_corrector_model.predict(corrector_input_features_filtered)
    corrected_joint_angles = joint_angles_seq - estimated_residuals

    # input_features = np.concatenate([corrected_joint_angles_seq, hand_angles_seq], axis=1)

    input_features = corrected_joint_angles
    n, d = input_features.shape

    # for min_cutoff in np.logspace(-5, 5, base=2):
    # for min_cutoff in np.arange(1):
    # for beta in np.logspace(-5, 5, base=2):
    #     print(min_cutoff)

    joint_angle_pca_coeffs = clutch_model.transform(clutch_scaler.transform(corrector_input_features_filtered[:, :15]))

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    for j in range(d):
        ax1.plot(input_features[:, j])
        ax2.plot(corrector_input_features_filtered[:, j])

    metric = "l1/2"
    pred_seqs = []

    # PCA
    pred_seq = []
    for i in range(n):
        clutch_distance = get_clutch_distance(joint_angle_pca_coeffs[i, :], clutch_coeffs,
                                              clutch_model.explained_variance_, metric)
        pred_seq.append(clutch_distance)
    pred_seqs.append(pred_seq)

    # just raw
    # pred_seq = []
    # for i in range(n):
    #     clutch_distance = get_clutch_distance(corrected_joint_angles[i, :], clutch_corrected_joint_angles,
    #                                           np.ones_like(corrected_joint_angles[i, :]), metric)
    #     pred_seq.append(clutch_distance)
    # pred_seqs.append(pred_seq)

    for pred_seq in pred_seqs:
        ax3.plot(pred_seq)

    # for ax, pred_seq in zip((ax3, ax4, ax5), pred_seqs):
    #     pred_seq = np.asarray(pred_seq)
        # pred_seq /= np.amax(pred_seq)
        # ax.plot(pred_seq)
    # ax3.plot(np.asarray(pred_seq) < CLUTCH_DIST_THRES)
    # plt.legend(["l1", "l2", "l1/2"])
    plt.show()


