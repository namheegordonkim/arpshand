import chumpy as ch
import numpy as np

from utils import MANO_TO_ZIMMERMAN_INDICES, BONES, ZIMMERMAN_TO_MANO_DICT
from visualize_skeleton import plot_mano_skeleton


def mano_inverse_kinematics(mano_hand, desired_zimmerman_coords, plot_progress=False):
    joint_idxs_mano = np.asarray(range(16))
    joint_idxs_zimmerman = MANO_TO_ZIMMERMAN_INDICES[joint_idxs_mano]
    assert (joint_idxs_zimmerman.shape == joint_idxs_mano.shape)

    joint_coords_mano = mano_hand.J_transformed[joint_idxs_mano]
    joint_coords_zimmerman = desired_zimmerman_coords[joint_idxs_zimmerman]

    mano_coords_centered = joint_coords_mano - joint_coords_mano[0]
    zimmerman_coords_centered = joint_coords_zimmerman - joint_coords_zimmerman[0]
    zimmerman_coords_centered *= 0.05

    if plot_progress:
        plot_mano_skeleton(mano_coords_centered, zimmerman_coords_centered)

    tip_idxs_mano = np.asarray([1, 4, 10, 7])  # knuckles (align palms)

    scale = ch.var(1)

    mano_coords_scaled = mano_coords_centered * scale

    position_loss = (mano_coords_scaled[tip_idxs_mano] - zimmerman_coords_centered[tip_idxs_mano]) ** 2

    loss = position_loss

    inputs = [mano_hand.pose[:3], scale]

    objective = {'loss': loss}

    opt = {'maxiter': 10}
    print("Rotating and scaling palms together...")
    ch.minimize(objective, x0=inputs, method='dogleg', options=opt)

    bone_idxs = []
    for zb0, zb1 in BONES:
        if zb0 in ZIMMERMAN_TO_MANO_DICT and zb1 in ZIMMERMAN_TO_MANO_DICT:
            b0 = ZIMMERMAN_TO_MANO_DICT[zb0]
            b1 = ZIMMERMAN_TO_MANO_DICT[zb1]
            bone_idxs.append([b0, b1])
    bone_idxs = np.asarray(bone_idxs)

    zimmerman_displacements = zimmerman_coords_centered[bone_idxs[:, 1]] - zimmerman_coords_centered[bone_idxs[:, 0]]
    mano_displacements = mano_coords_scaled[bone_idxs[:, 1]] - mano_coords_scaled[bone_idxs[:, 0]]

    displacement_loss = ch.sum(ch.multiply(zimmerman_displacements, mano_displacements))

    regularization_loss = 1e-5 * ch.sum(mano_hand.pose[3:] ** 2)
    # MAGIC HAPPENING RIGHT HERE:
    # This huge number is important.
    # Apparently, ch.minimize will attempt to push the loss to zero,
    # not towards minus infinity.
    # Here we're trying to maximize the dot product of all bones
    # in both hands (thereby making them all point in the same
    # direction). The dot product needs to be maximally positive,
    # and should not be zero, hence the huge number below.
    loss = (1000000 - displacement_loss) + regularization_loss
    inputs = [mano_hand.pose[3:]]
    opt = {'maxiter': 10}
    objective = {'loss': loss}
    print("Aligning all bones to be co-linear...")
    ch.minimize(objective, x0=inputs, method='dogleg', options=opt)

    if plot_progress:
        plot_mano_skeleton(mano_coords_scaled, zimmerman_coords_centered)
