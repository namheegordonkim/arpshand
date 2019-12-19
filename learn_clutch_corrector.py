import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

from utils import get_clutch_data, get_reference_joint_angles


def main():
    # load the reference pose
    ref_joint_angles = get_reference_joint_angles()

    # load all clutch poses
    hand_angles_seq, joint_angles_seq, root_poses_seq = get_clutch_data()
    # hand_angles_seq, joint_angles_seq, root_poses_seq = get_all_dynamic_data()

    # generate residuals
    joint_angle_residual_seq = joint_angles_seq - ref_joint_angles

    # regress the residuals with hand angles
    gamma = 1e-4
    alpha = 1e-2
    # for alpha in np.logspace(-5, 5, base=10):
    #     print(alpha)
    scaler = StandardScaler()
    model = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
    corrector_input_features = np.concatenate([joint_angles_seq, hand_angles_seq, root_poses_seq], axis=1)
    # corrector_input_features = np.concatenate([hand_angles_seq, root_poses_seq], axis=1)
    corrector_input_features = scaler.fit_transform(corrector_input_features)
    model.fit(corrector_input_features, joint_angle_residual_seq)

    estimated_residuals = model.predict(corrector_input_features)
    corrected_joint_angles_seq = joint_angles_seq - estimated_residuals
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
    for j in range(15):
        ax1.plot(joint_angles_seq[:, j])
        ax2.plot(joint_angle_residual_seq[:, j])
        ax3.plot(corrected_joint_angles_seq[:, j])
    for j in range(3):
        ax4.plot(hand_angles_seq[:, j])
        # ax4.plot(estimated_residuals[:, j])
    plt.show()

    with open("./models/clutch_corrector.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("./models/clutch_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
