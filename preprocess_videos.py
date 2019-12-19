import argparse
import glob
import os

import imageio
import numpy as np
from pathlib import Path
DATA_DIR = "./data"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    mim = np.stack(imageio.mimread(input_file))[::2, ...]
    print(mim.shape)
    n_frames, height, width, channels = mim.shape
    for i in range(n_frames):
        imageio.imwrite("{:s}/{:06d}.jpg".format(str(output_dir), i), mim[i])

