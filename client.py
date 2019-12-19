import socket
from collections import deque

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import zlib
import time
from multiprocessing import Process

from PIL import Image
from sklearn.cluster import KMeans

from utils import convert_coords_to_angles, get_rotation_basis, get_clutch_distance, detect_clutch, predict_gesture, \
    compress_and_send_to_socket, receive_from_socket
from pydub import AudioSegment
from pydub.playback import play

from oneeuro import OneEuroFilter

GESTURE_THRES = 1.1
MAX_FPS = 15


def play_sound(sound: AudioSegment):
    play(sound)


def main():
    fig, ((im_ax, all_angle_ax), (clutch_distance_ax, gesture_angle_ax)) = plt.subplots(nrows=2, ncols=2)
    # ax = plt.axes(projection='3d')

    with open("./data/clutch.pkl", "rb") as f:
        clutch_root_poses, clutch_hand_angles, clutch_joint_angles = pickle.load(f)
    with open("./data/clutch_pca_model.pkl", "rb") as f:
        clutch_model = pickle.load(f)
    with open("./data/clutch_pca_coeffs.pkl", "rb") as f:
        clutch_coeffs = pickle.load(f)
    with open("./models/clutch_corrector.pkl", "rb") as f:
        clutch_corrector_model = pickle.load(f)
    with open("./models/clutch_scaler.pkl", "rb") as f:
        clutch_scaler = pickle.load(f)

    clustering_model = KMeans(n_clusters=30, n_init=100)
    clustering_model.fit(clutch_coeffs)
    cluster_centers = clustering_model.cluster_centers_

    # beep sound
    beep = AudioSegment.from_wav("./beep.wav")
    confirm = AudioSegment.from_wav("./confirm.wav")

    plt.ion()
    plt.show()
    # app.run()

    # canvas = scene.SceneCanvas(keys='interactive')
    # canvas.size = 800, 600
    # canvas.show()

    cap = cv2.VideoCapture(4)
    # flush first few frames while camera boots up
    for _ in range(50):
        ret, image_raw = cap.read()
        cv2.imshow("frame", image_raw)
        cv2.waitKey(1)
        assert ret
    cv2.destroyAllWindows()

    HOST = '127.0.0.1'
    PORT = 3333

    # load state machine variables
    all_angles_buffer = deque(maxlen=1024)
    clutch_distance_buffer = deque(maxlen=8)
    all_clutch_distance_buffer = deque(maxlen=1024)
    gesture_features_buffer = deque(maxlen=128)
    clutch_buffer = deque(maxlen=8)
    clutch_engaged = False
    clutch_started = False
    first = True

    # load classifier
    with open("./models/onedollar.pkl", "rb") as f:
        gesture_classifier = pickle.load(f)

    # initialize a 1-euro filter per feature
    min_cutoff = 0.45931347703523806
    beta = 0.2
    filters = [OneEuroFilter(13, min_cutoff, beta, 0.1) for _ in range(18)]
    message = ""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setblocking(True)
        s.connect((HOST, PORT))
        s2c_data_remaining = b""
        # try:
        while True:
            ret, image_raw = cap.read()
            assert ret
            image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_raw)
            image_resized = np.asarray(image.resize((320, 240)))
            # image_resized = resize(image_raw, (240, 320), order=3)

            # Send
            # print("Client: send image")
            compress_and_send_to_socket(image_resized, s)
            # print("Client: sent image")

            s2c_data_trimmed, s2c_data_remaining = receive_from_socket(s, s2c_data_remaining)
            toc = time.time()
            # print("Send-Recv elapsed time:\t{:f}".format(toc-tic))
            tic = time.time()
            s2c_data_decompressed = zlib.decompress(s2c_data_trimmed)
            recvtuple = pickle.loads(s2c_data_decompressed)
            scale_v, center_v, keypoint_coord3d_v, scoremap_nonzeros = recvtuple
            hand_present_yes = scoremap_nonzeros > 2000

            # print(keypoint_coord3d_v)
            root_pose, hand_angles, data_joint_angles = convert_coords_to_angles(keypoint_coord3d_v, scale_v, center_v)

            raw_input_features = np.concatenate([data_joint_angles, hand_angles, root_pose])

            # apply filtering
            # filtered_input_features = np.asarray(
            #     [filter.filter(angle) for (filter, angle) in zip(filters, raw_input_features)])
            filtered_input_features = raw_input_features
            estimated_residuals = clutch_corrector_model.predict(filtered_input_features.reshape(1, -1)).reshape(-1)
            corrected_joint_angles = data_joint_angles - estimated_residuals
            gesture_input_features = np.concatenate([data_joint_angles, hand_angles])

            all_angles_buffer.append(gesture_input_features)
            basis = get_rotation_basis(keypoint_coord3d_v)
            coords_to_plot = keypoint_coord3d_v @ basis
            input_joint_angle_pca_coeffs = clutch_model.transform(
                clutch_scaler.transform(corrected_joint_angles.reshape(1, -1))).reshape(-1)

            # clutch_distance = get_clutch_distance(filtered_input_features, clutch_joint_angles)
            clutch_distance = get_clutch_distance(input_joint_angle_pca_coeffs, cluster_centers,
                                                  clutch_model.explained_variance_, "l1/2")
            # cosine-distance
            # distance = (data_joint_angles @ mean_clutch_joint_angles) / \
            #            (np.linalg.norm(data_joint_angles) * np.linalg.norm(mean_clutch_joint_angles))
            clutch_distance_buffer.append(clutch_distance)
            all_clutch_distance_buffer.append(clutch_distance)
            clutch_yes = detect_clutch(clutch_distance_buffer)
            print(clutch_distance, clutch_yes, clutch_started, clutch_engaged, hand_present_yes)

            # engage clutch if average distance in the past few poses is small
            p_beep = Process(target=play_sound, args=[beep])
            p_confirm = Process(target=play_sound, args=[confirm])
            if not hand_present_yes or first:
                # reset the state machine
                # clutch_engaged = False
                # clutch_started = False
                # first = True
                # all_angles_buffer.clear()
                # clutch_distance_buffer.clear()
                # all_clutch_distance_buffer.clear()
                # gesture_features_buffer.clear()
                message = "NO HAND"
            if hand_present_yes:
                if clutch_yes:
                    if clutch_engaged and not clutch_started:
                        clutch_engaged = False
                        clutch_distance_buffer.clear()
                        if len(gesture_features_buffer) > 10:
                            for _ in range(10):
                                if len(gesture_features_buffer) > 0:
                                    gesture_features_buffer.pop()
                        pred, dist = predict_gesture(gesture_classifier, gesture_features_buffer)
                        gesture_features_buffer.clear()
                        p_confirm.start()
                        print(pred)
                        message = "{:s}:\t{:f}".format(pred, dist)
                        continue

                    if not clutch_started and not clutch_engaged:
                        clutch_started = True
                        p_beep.start()
                        message = "CLUTCH STARTED"
                        clutch_buffer.append(gesture_input_features)
                else:
                    if clutch_started:
                        clutch_distance_buffer.clear()
                        gesture_features_buffer.clear()
                        clutch_started = False
                        clutch_engaged = True
                        message = "CLUTCH ENGAGED"

                if clutch_engaged:
                    gesture_features_buffer.append(gesture_input_features)
                #     if len(gesture_features_buffer) > 5:
                #         gesture_classifier_input = list(clutch_buffer)
                #         gesture_classifier_input.extend(gesture_features_buffer)
                #         pred, dist = predict_gesture(gesture_classifier, gesture_classifier_input)
                #         print(pred, dist)
                #         message = "{:s}:\t{:f}".format(pred, dist)
                #         if dist < GESTURE_THRES:
                #             p_confirm.start()
                #             stationary_angles = np.mean(np.stack(gesture_features_buffer), axis=0)
                #             gesture_features_buffer.clear()
                #             clutch_engaged = False

            toc = time.time()
            # print("FSM update elapsed time:\t{:f}".format(toc-tic))

            tic = time.time()
            all_angle_ax.cla()
            all_angle_ax.set_ylim(1, 4)
            all_angles_np = np.stack(all_angles_buffer)
            _, n_joints = all_angles_np.shape
            for j in range(n_joints):
                all_angle_ax.plot(all_angles_np[:, j])
            #
            # clutch_distance_ax.cla()
            # clutch_distance_ax.plot(np.asarray(all_clutch_distance_buffer))

            # gesture_angle_ax.cla()
            # gesture_angle_ax.set_ylim(1, 4)
            # if len(gesture_features_buffer) > 0:
            #     gesture_angles_np = np.stack(gesture_features_buffer)
            #     for j in range(n_joints):
            #         gesture_angle_ax.plot(gesture_angles_np[:, j])

            # ax.cla()
            # ax.set_xlim3d(-2, 2)
            # ax.set_ylim3d(2, -2)
            # ax.set_zlim3d(2, -2)
            # plot_hand_3d(coords_to_plot, ax)
            # gesture_angle_ax.imshow(argmax_scoremap)

            plt.draw()
            plt.pause(0.01)
            toc = time.time()
            # print("Visual elapsed time:\t{:f}".format(toc-tic))
            cv2.putText(image_raw, message, (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
            image_to_show = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", image_to_show)
            cv2.waitKey(1)
            first = False
        # except Exception:
        #     pass

    print("Client: closed")


if __name__ == "__main__":
    main()
