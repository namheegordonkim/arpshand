'''
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de

Acknowledgements:
The code file is based on the release code of http://smpl.is.tue.mpg.de with adaptations.
Therefore, we would like to kindly thank Matthew Loper and Naureen Mahmood.


Please Note:
============
This is a demo version of the script for driving the MANO model with python.
We would be happy to receive comments, help and suggestions on improving this code
and in making it available on more platforms.


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]
- OpenCV [http://opencv.org/downloads.html]
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the smpl model and rendering it using OpenDR
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below).

This code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the mano/webuser/hello_world directory. You can run
the hello world script now by typing the following:
>	python MANO___render.py


'''
import argparse
import pickle

import numpy as np
from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers
from psbody.mesh.sphere import Sphere

from mano_inverse_kinematics import mano_inverse_kinematics
from mano_v1_2.webuser.smpl_handpca_wrapper_HAND_only import load_model
from utils import get_reference_joint_coords, \
    convert_coords_to_rotationally_invariant_displacements, \
    convert_rotationally_invariant_displacements_to_coords, load_coords, MANO_TO_ZIMMERMAN_INDICES
from visualize_skeleton import plot_mano_skeleton


class MANOWrapper:
    """
    Custom wrapper to interact with MANO model more easily
    """

    def __init__(self, m):
        self.m = m
        self.m.betas[:] = np.random.rand(m.betas.size) * .3
        # m.pose[:] = np.random.rand(m.pose.size) * .2
        self.m.pose[:3] = [0., 0., 0.]
        self.m.pose[3:] = np.zeros(45)
        # m.pose[3:] = [-0.42671473, -0.85829819, -0.50662164, +1.97374622, -0.84298473, -1.29958491]
        self.m.pose[0] = np.pi

        # compute inverse components to map from fullpose spec to coefficients
        hands_components = np.asarray(m.hands_components)
        self.hands_components_inv = np.linalg.inv(hands_components)

        # rendering components
        # Assign attributes to renderer
        w, h = (640, 480)

        # Create OpenDR renderer
        self.rn = ColoredRenderer()
        self.rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([-0.03, -0.04, 0.20]), f=np.array([w, w]) / 2.,
                                       c=np.array([w, h]) / 2., k=np.zeros(5))
        self.rn.frustum = {'near': 0.01, 'far': 2., 'width': w, 'height': h}
        self.rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

        # Construct point light source
        self.rn.vc = LambertianPointLight(f=m.f,
                                          v=self.rn.v,
                                          num_verts=len(m),
                                          light_pos=np.array([-1000, -1000, -2000]),
                                          vc=np.ones_like(m) * .9,
                                          light_color=np.array([1., 1., 1.]))
        self.rn.vc += LambertianPointLight(f=m.f,
                                           v=self.rn.v,
                                           num_verts=len(m),
                                           light_pos=np.array([+2000, +2000, +2000]),
                                           vc=np.ones_like(m) * .9,
                                           light_color=np.array([1., 1., 1.]))
        self.mvs = MeshViewers(window_width=2000, window_height=800, shape=[1, 3])

    def set_hand_rotation(self, hand_angles):
        """
        Set the hand rotation.
        """
        self.m.pose[:3] = hand_angles

    def set_joint_rotation(self, joint_angles):
        """
        Set the joint rotation.
        """
        coeffs = joint_angles.reshape(-1, ) @ self.hands_components_inv
        self.m.pose[3:] = coeffs

    def get_hand_rotation(self):
        """
        Get the hand rotation.
        """
        return self.m.fullpose[:3]

    def get_joint_rotation(self):
        """
        Get the joint rotation.
        """
        return self.m.fullpose[3:].reshape(15, 3)

    def render(self):
        radius = .01
        model_Mesh = Mesh(v=self.m.r, f=self.m.f)
        model_Joints = [Sphere(np.array(jointPos), radius).to_mesh(np.eye(3)[0 if jointID == 0 else 1]) for
                        jointID, jointPos in enumerate(self.m.J_transformed)]
        self.mvs[0][0].set_static_meshes([model_Mesh] + model_Joints, blocking=True)
        self.mvs[0][1].set_static_meshes([model_Mesh], blocking=True)
        model_Mesh = Mesh(v=self.m.r, f=[])
        self.mvs[0][2].set_static_meshes([model_Mesh] + model_Joints, blocking=True)


def plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    for joint_idx in range(coords_xyz.shape[0]):
        axis.text(coords_xyz[joint_idx, 0], coords_xyz[joint_idx, 1], coords_xyz[joint_idx, 2], str(joint_idx))
    axis.view_init(azim=-90., elev=90.)


def get_rotation_basis(triangle):
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


def main():
    """
    Animate the MANO hand model with a synthetic hand pose trajectory.
    Takes a while (~8 seconds per frame) to run inverse kinematics.
    """

    # Load MANO model (here we load the right hand model)
    m = load_model('mano_v1_2/models/MANO_LEFT.pkl', ncomps=45, flat_hand_mean=True)
    hand: MANOWrapper = MANOWrapper(m)

    gesture_name = args.gesture_name
    speed = args.speed
    with open("./data/{:s}_generated_{:.2f}.pkl".format(gesture_name, speed), "rb") as f:
        reference_disps_seq = pickle.load(f)

    mano_pose_seq = []

    n = len(reference_disps_seq)
    for i in range(n):
        print((i + 1), " / ", n)
        displacements = reference_disps_seq[i]
        recovered_coords = convert_rotationally_invariant_displacements_to_coords(displacements)
        mano_inverse_kinematics(hand.m, recovered_coords)
        pose = np.copy(hand.m.pose)
        mano_pose_seq.append(pose)
    mano_pose_seq = np.asarray(mano_pose_seq)

    # Save pose sequence to pickle file
    file_name = "tmp.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(mano_pose_seq, f)

    with open(file_name, "rb") as f:
        mano_pose_seq = pickle.load(f)

    animation_index = 0
    while True:
        for i in range(len(hand.m.pose)):
            hand.m.pose[i] = mano_pose_seq[animation_index, i]
        hand.render()
        animation_index = (animation_index + 1) % len(mano_pose_seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture_name", type=str, required=True)
    parser.add_argument("--speed", type=float, required=True)

    args = parser.parse_args()

    main()
