import socket
import tensorflow as tf
import numpy as np
import cv2
from hand3d.nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
import pickle
import zlib


def main():
    HOST = '127.0.0.1'
    PORT = 3333

    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, \
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print("Server: listening...")
        while True:
            print("Server: looking for connections")
            conn, addr = s.accept()
            print("Server: connected by", addr)
            c2s_data_remaining = b""
            while True:
                c2s_data_trimmed, c2s_data_remaining, should_break = receive_from_connection(conn, c2s_data_remaining)
                if should_break:
                    break
                # print("Server: received data_received: ", repr(data_received))
                c2s_data_decompressed = zlib.decompress(c2s_data_trimmed)
                im = np.fromstring(c2s_data_decompressed, dtype=np.uint8)
                image_resized = cv2.imdecode(im, cv2.IMREAD_COLOR)
                image_v = np.expand_dims((image_resized.astype('float') / 255.0) - 0.5, 0)

                hand_scoremap_v, image_crop_v, scale_v, center_v, \
                keypoints_scoremap_v, keypoint_coord3d_v = sess.run(
                    [hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                     keypoints_scoremap_tf, keypoint_coord3d_tf],
                    feed_dict={image_tf: image_v})

                keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)
                # sendtuple = (hand_scoremap_v, image_crop_v, scale_v, center_v, keypoints_scoremap_v, keypoint_coord3d_v)
                hand_scoremap_v = np.squeeze(hand_scoremap_v)
                argmax_scoremap = np.argmax(hand_scoremap_v, 2)
                scoremap_nonzeros = np.sum(argmax_scoremap > 0)

                sendtuple = (scale_v, center_v, keypoint_coord3d_v, scoremap_nonzeros)
                senddata = pickle.dumps(sendtuple)
                senddata_compressed = zlib.compress(senddata)
                senddata_delimited = b"<START>" + senddata_compressed + b"<END>"
                conn.sendall(senddata_delimited)


def receive_from_connection(connection, initial_data=b""):
    c2s_data = initial_data
    # print("Client: start blocking")
    should_break = False
    while b"<END>" not in c2s_data:
        received = connection.recv(1024)
        c2s_data += received
        if not received:
            c2s_data = b""
            print("Server: received end of transmission")
            should_break = True
            return b"", b"", should_break
        # print("Client: receive chunk")
    c2s_data_trimmed = c2s_data.split(b"<START>", maxsplit=1)[1]
    c2s_data_trimmed = c2s_data_trimmed.split(b"<END>", maxsplit=1)[0]
    c2s_data_remaining = c2s_data.split(b"<END>", maxsplit=1)[1]
    print("Server: received message.")
    return c2s_data_trimmed, c2s_data_remaining, should_break


if __name__ == "__main__":
    main()
