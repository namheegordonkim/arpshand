import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import get_clutch_data


def main():
    with open("./models/clutch_corrector.pkl", "rb") as f:
        clutch_corrector_model = pickle.load(f)
    with open("./models/clutch_scaler.pkl", "rb") as f:
        clutch_scaler = pickle.load(f)

    hand_angles_seq, joint_angles_seq, root_poses_seq = get_clutch_data()
    # disps_seq = []
    # for clutch_filename in clutch_filenames:
    #     with open(clutch_filename, "rb") as f:
    #         disps = pickle.load(f)
    #     disps_seq.append(disps)

    root_poses_seq = np.stack(root_poses_seq)
    hand_angles_seq = np.stack(hand_angles_seq)
    joint_angles_seq = np.stack(joint_angles_seq)
    # disps_seq = np.stack(disps_seq)

    # root_pose_mean = np.mean(root_poses_seq, axis=0)
    # hand_angles_mean = np.mean(hand_angles_seq, axis=0)
    # joint_angles_mean = np.mean(joint_angles_seq4, axis=0)

    input_features = np.concatenate([joint_angles_seq, hand_angles_seq, root_poses_seq], axis=1)
    input_features = clutch_scaler.transform(input_features)
    estimated_joint_angle_residuals_seq = clutch_corrector_model.predict(input_features)
    corrected_joint_angles_seq = joint_angles_seq - estimated_joint_angle_residuals_seq

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    for j in range(15):
        ax1.plot(joint_angles_seq[:, j])
        ax2.plot(corrected_joint_angles_seq[:, j])
    plt.show()

    with open("./data/clutch.pkl", "wb") as f:
        pickle.dump((root_poses_seq, hand_angles_seq, corrected_joint_angles_seq), f)



if __name__ == "__main__":
    main()
