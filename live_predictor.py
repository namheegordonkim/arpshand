import pickle
import socket
import zlib
from collections import deque
from multiprocessing import Process

import cv2
import numpy as np
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play

from utils import compress_and_send_to_socket, convert_coords_to_angles, receive_from_socket, EXTREMETY_IDX

NO_HAND = "no_hand"
IDLE_STATE = "idle"
CLUTCH_STATE = "clutch"
RECORDING_STATE = "recording"


def get_next_pose(image_raw, socket, s2c_data_remaining):
    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_raw)
    image_resized = np.asarray(image.resize((320, 240)))

    # Send
    compress_and_send_to_socket(image_resized, socket)

    # Receive
    s2c_data_trimmed, s2c_data_remaining = receive_from_socket(socket, s2c_data_remaining)
    s2c_data_decompressed = zlib.decompress(s2c_data_trimmed)
    recvtuple = pickle.loads(s2c_data_decompressed)
    scale_v, center_v, keypoint_coord3d_v, scoremap_nonzeros = recvtuple
    hand_present_yes = scoremap_nonzeros > 1000
    root_pose, hand_angles, data_joint_angles = convert_coords_to_angles(keypoint_coord3d_v, scale_v, center_v)
    return root_pose, hand_angles, data_joint_angles, hand_present_yes


def get_clutch_distance(current_joint_angles, clutch_poses):
    """
    Given the current joint angles and the database of clutch joint angles,
    return the distance between the current pose and the clutch
    """
    residuals = clutch_poses - current_joint_angles
    # remove tip angles for better performance
    residuals = np.delete(residuals, EXTREMETY_IDX, axis=1)
    distances = np.sum(np.abs(residuals) ** (1. / 2), axis=1) ** 2
    distance = np.min(distances)
    return distance


def detect_clutch(clutch_distance_buffer):
    avg_distance = np.median(clutch_distance_buffer)
    std = np.std(clutch_distance_buffer)
    dist_good_yes = avg_distance < 0.25
    buffer_full_yes = len(clutch_distance_buffer) == clutch_distance_buffer.maxlen
    std_good_yes = std < 1.0
    return dist_good_yes and buffer_full_yes and std_good_yes


def classify(gesture_classifier, gesture_feature_buffer, clutch_buffer):
    gesture_features = np.concatenate([np.stack(clutch_buffer), np.stack(gesture_feature_buffer)], axis=0)
    # throw away tip angles for improved performance
    gesture_features = np.delete(gesture_features, EXTREMETY_IDX, axis=1)
    label, dist = gesture_classifier.predict_gesture(gesture_features)
    return label, dist


def play_sound(sound: AudioSegment):
    play(sound)


MAX_BUFFER_LENGTH = 30


def main():
    # Loading all models and data

    # load clutch-related models and variables
    with open("./data/clutch.pkl", "rb") as f:
        clutch_root_poses, clutch_hand_angles, clutch_joint_angles = pickle.load(f)
    clutch_poses = clutch_joint_angles
    with open("./data/clutch_pca_model.pkl", "rb") as f:
        clutch_model = pickle.load(f)
    with open("./data/clutch_pca_coeffs.pkl", "rb") as f:
        clutch_coeffs = pickle.load(f)
    with open("./models/clutch_corrector.pkl", "rb") as f:
        clutch_corrector_model = pickle.load(f)
    with open("./models/clutch_scaler.pkl", "rb") as f:
        clutch_scaler = pickle.load(f)

    # load classifier
    with open("./models/onedollar.pkl", "rb") as f:
        gesture_classifier = pickle.load(f)

    # beep sound
    beep = AudioSegment.from_wav("./beep.wav")
    confirm = AudioSegment.from_wav("./confirm.wav")

    cap = cv2.VideoCapture(0)
    # flush first few frames while camera boots up
    for _ in range(50):
        ret, image_raw = cap.read()
        cv2.imshow("frame", image_raw)
        cv2.waitKey(1)
        assert ret
    cv2.destroyAllWindows()

    current_state = NO_HAND
    clutch_buffer = deque(maxlen=8)
    clutch_distance_buffer = deque(maxlen=8)
    pose_buffer = deque(maxlen=128)
    was_in_clutch_pose = False
    is_in_clutch_pose = False

    HOST = '127.0.0.1'
    PORT = 3333
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setblocking(True)
        s.connect((HOST, PORT))
        s2c_data_remaining = b""
        pred_message = ""
        while True:
            # Get input from camera
            ret, image_raw = cap.read()

            # Sound preprocessing
            p_beep = Process(target=play_sound, args=[beep])
            p_confirm = Process(target=play_sound, args=[confirm])

            # General updates
            # Retrieve the next pose from server
            root_pose, hand_angles, data_joint_angles, hand_present_yes = get_next_pose(image_raw, s,
                                                                                        s2c_data_remaining)

            # Preprocess the next pose
            corrector_input_features = np.concatenate([data_joint_angles, hand_angles, root_pose])
            corrector_input_features = clutch_scaler.transform(corrector_input_features.reshape(1, -1)).reshape(-1)
            joint_angle_residuals = clutch_corrector_model.predict(corrector_input_features.reshape(1, -1)).reshape(-1)
            corrected_joint_angles = data_joint_angles - joint_angle_residuals

            # Get distance between current pose and clutch poses
            clutch_distance = get_clutch_distance(corrected_joint_angles, clutch_poses)
            clutch_distance_buffer.append(clutch_distance)
            for i in range(len(clutch_distance_buffer)):
                clutch_distance_buffer[i] *= 1.02

            # Preprocess features for gesture classifier
            # current_pose = np.concatenate([corrected_joint_angles, hand_angles])
            current_pose = np.concatenate([data_joint_angles, hand_angles])
            # current_pose = np.concatenate([joint_angle_residuals, hand_angles])
            clutch_buffer.append(current_pose)

            # State machine utility
            was_in_clutch_pose = is_in_clutch_pose

            # Detect whether current pose is a clutch pose
            is_in_clutch_pose = detect_clutch(clutch_distance_buffer)

            print(clutch_distance_buffer)
            print(current_state, np.median(clutch_distance))

            if not hand_present_yes:
                current_state = NO_HAND

            # State-specific logic
            state_message = ""
            if current_state == NO_HAND:
                state_message = "NO HAND"
                clutch_buffer.clear()
                clutch_distance_buffer.clear()
                if hand_present_yes:
                    current_state = IDLE_STATE

            elif current_state == IDLE_STATE:
                state_message = "IDLE"
                # Waiting for clutch
                if is_in_clutch_pose and not was_in_clutch_pose:
                    current_state = CLUTCH_STATE
                    p_beep.start()

            elif current_state == CLUTCH_STATE:
                state_message = "CLUTCH"
                # Waiting for clutch to end to start recording
                if not is_in_clutch_pose:
                    current_state = RECORDING_STATE
                    pose_buffer.clear()

            elif current_state == RECORDING_STATE:
                state_message = "RECORDING"

                # Recording pose data
                pose_buffer.append(current_pose)

                # Attempt to classify
                if len(pose_buffer) > 8:
                    label, dist = classify(gesture_classifier, pose_buffer, clutch_buffer)
                    pred_message = "{:s}:{:03f}".format(label, dist)
                    if is_in_clutch_pose:
                        current_state = IDLE_STATE
                        p_confirm.start()

            else:
                raise Exception("Invalid state")

            # always render a frame
            # image_to_show = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            cv2.putText(image_raw, state_message, (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(image_raw, pred_message, (320, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            image_to_show = image_raw
            cv2.imshow("frame", image_to_show)
            cv2.waitKey(1)

        pass


if __name__ == "__main__":
    main()
