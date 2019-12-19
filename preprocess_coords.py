import glob
import os
import pickle

import argparse
import numpy as np
from scipy.spatial.transform import Rotation

from pathlib import Path
from utils import get_angle, TRIPLETS, convert_coords_to_angles

DATA_DIR = "./data"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    coords_filenames = sorted(glob.glob("{:s}/*.pkl".format(input_dir), recursive=True))
    for coords_filename in coords_filenames:
        basename = os.path.basename(coords_filename)
        filename_nums = basename.split("_")[0]
        joint_angles = np.zeros(TRIPLETS.shape[0])
        with open(coords_filename, "rb") as f:
            coords, scale, center = pickle.load(f)
        root_pose, hand_angles, joint_angles = convert_coords_to_angles(coords, scale, center)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        save_filename = "{:s}/{:s}_angles.pkl".format(output_dir, filename_nums)
        with open(save_filename, "wb") as f:
            pickle.dump((root_pose, hand_angles, joint_angles), f)
