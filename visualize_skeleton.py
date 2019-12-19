import matplotlib.pyplot as pp
import numpy as np

from utils import BONES, ZIMMERMAN_TO_MANO_DICT


def plot_mano_skeleton(hand_coords_1, hand_coords_2=None):
    fig = pp.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(hand_coords_1[0, 0], hand_coords_1[0, 1], hand_coords_1[0, 2], marker='o',
               color="r")
    ax.scatter(hand_coords_1[1:, 0], hand_coords_1[1:, 1], hand_coords_1[1:, 2], marker='o')

    if hand_coords_2 is not None:
        ax.scatter(hand_coords_2[0, 0], hand_coords_2[0, 1], hand_coords_2[0, 2],
                   marker='^', color="r")
        ax.scatter(hand_coords_2[1:, 0], hand_coords_2[1:, 1], hand_coords_2[1:, 2],
                   marker='^')

    for idx0, idx1 in BONES:
        if idx0 in ZIMMERMAN_TO_MANO_DICT and idx1 in ZIMMERMAN_TO_MANO_DICT:
            idx0_mano = ZIMMERMAN_TO_MANO_DICT[idx0]
            idx1_mano = ZIMMERMAN_TO_MANO_DICT[idx1]

            coords0 = hand_coords_1[idx0_mano]
            coords1 = hand_coords_1[idx1_mano]
            coords = np.stack([coords0, coords1])
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])

    if hand_coords_2 is not None:
        for idx0, idx1 in BONES:
            if idx0 in ZIMMERMAN_TO_MANO_DICT and idx1 in ZIMMERMAN_TO_MANO_DICT:
                idx0_mano = ZIMMERMAN_TO_MANO_DICT[idx0]
                idx1_mano = ZIMMERMAN_TO_MANO_DICT[idx1]
                coords0 = hand_coords_2[idx0_mano]
                coords1 = hand_coords_2[idx1_mano]
                coords = np.stack([coords0, coords1])
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    pp.show()
