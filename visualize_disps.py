import argparse
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

from oneeuro import OneEuroFilter
from utils import get_clutch_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disps_dir", required=True, type=str)

    args = parser.parse_args()
    disps_dir = args.disps_dir

    disps_seq = []
    disps_filenames = sorted(glob.glob("{:s}/*.pkl".format(disps_dir)))
    for disps_filename in disps_filenames:
        with open(disps_filename, 'rb') as f:
            disps = pickle.load(f)
        disps_seq.append(disps)
    disps_seq = np.stack(disps_seq)
    n, d = disps_seq.shape

    plt.figure()
    for j in range(d):
        plt.plot(disps_seq[:, j])
    plt.show()


