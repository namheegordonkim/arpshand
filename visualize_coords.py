import argparse
import glob
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hand3d.utils.general import plot_hand_3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords_path", required=True, type=str)

    args = parser.parse_args()
    coords_path = args.coords_path
    coords_filenames = glob.glob("{:s}/*.pkl".format(coords_path))
    plt.figure()
    ax = plt.axes(projection='3d')
    plt.ion()
    for coords_filename in coords_filenames:
        with open(coords_filename, 'rb') as f:
            scale, center, coords = pickle.load(f)
            ax.cla()
            plot_hand_3d(coords, ax)
            plt.draw()
            plt.pause(0.01)

