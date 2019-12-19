import argparse
import glob
import os
import pickle
import socket
import zlib
import pathlib

import imageio
import numpy as np
from pathlib import Path

from PIL import Image

from utils import compress_and_send_to_socket, receive_from_socket, TRIPLETS, convert_coords_to_angles, \
    convert_coords_to_rotationally_invariant_displacements

DATA_DIR = "./data"

HOST = '127.0.0.1'
PORT = 3333


def resize(im_ndarray):
    image = Image.fromarray(im_ndarray)
    image_resized = np.asarray(image.resize((320, 240)))
    return image_resized


def main():
    mim = np.stack(imageio.mimread(input_file, memtest=False))[::2, ...]
    mim_resized = np.stack([resize(im) for im in mim])
    print(mim_resized.shape)
    s2c_data_remaining = b""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setblocking(True)
        s.connect((HOST, PORT))
        output_list = []
        for im_resized in mim_resized:
            compress_and_send_to_socket(im_resized, s)
            s2c_data_trimmed, s2c_data_remaining = receive_from_socket(s, s2c_data_remaining)
            s2c_data_decompressed = zlib.decompress(s2c_data_trimmed)
            recvtuple = pickle.loads(s2c_data_decompressed)
            output_list.append(recvtuple)
        print(output_list)
        print(len(output_list))
        for i, recvtuple in enumerate(output_list):
            scale_v, center_v, keypoint_coord3d_v = recvtuple
            # root_pose, hand_angles, joint_angles = convert_coords_to_angles(keypoint_coord3d_v, scale_v, center_v)
            disps = convert_coords_to_rotationally_invariant_displacements(keypoint_coord3d_v)
            save_filename = "{:s}/{:06d}_disps.pkl".format(output_dir, i)
            with open(save_filename, "wb") as f:
                # pickle.dump((root_pose, hand_angles, joint_angles), f)
                pickle.dump(disps, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    main()
