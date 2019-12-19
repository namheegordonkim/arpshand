import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from onedollarrecognizer import OneDollarRecognizer
from utils import load_gesture_lists, EXTREMETY_IDX, get_all_dynamic_data


def register_gesture(gesture_classifier, clutch_corrector_model, gesture_name, index_range):
    for i in index_range:
        joint_angles_list, _, hand_angles_list, root_pose_list = load_gesture_lists(gesture_name, [i])

        joint_angles = joint_angles_list[0]
        hand_angles = hand_angles_list[0]
        root_poses = root_pose_list[0]

        # data augmentation: apply mirroring
        joint_angles = np.concatenate([joint_angles, joint_angles[::-1, :]])
        hand_angles = np.concatenate([hand_angles, hand_angles[::-1, :]])
        root_poses = np.concatenate([root_poses, root_poses[::-1, :]])

        # compute residual corrections
        # corrector_input_features = np.concatenate([joint_angles, hand_angles, root_poses], axis=1)
        # joint_angles_residuals = clutch_corrector_model.predict(corrector_input_features)
        # corrected_joint_angles = joint_angles - joint_angles_residuals

        # cut off tip angles for better performance
        # corrected_joint_angles = np.delete(corrected_joint_angles, EXTREMETY_IDX, axis=1)
        joint_angles = np.delete(joint_angles, EXTREMETY_IDX, axis=1)
        # joint_angles_residuals = np.delete(joint_angles_residuals, EXTREMETY_IDX, axis=1)

        # gesture_input = np.concatenate([corrected_joint_angles, hand_angles], axis=1)
        # gesture_input = np.concatenate([joint_angles_residuals, hand_angles], axis=1)
        gesture_input = np.concatenate([joint_angles, hand_angles], axis=1)
        # gesture_input = gesture_scaler.transform(gesture_input)
        gesture_classifier.define_gesture(gesture_input, gesture_name)


def main():
    with open("./models/clutch_corrector.pkl", "rb") as f:
        clutch_corrector_model = pickle.load(f)

    gesture_classifier = OneDollarRecognizer()
    register_gesture(gesture_classifier, clutch_corrector_model, "scissors", np.arange(3))
    register_gesture(gesture_classifier, clutch_corrector_model, "paper", np.arange(3))
    register_gesture(gesture_classifier, clutch_corrector_model, "ok", np.arange(7))
    register_gesture(gesture_classifier, clutch_corrector_model, "thumbs_up", np.arange(3))
    register_gesture(gesture_classifier, clutch_corrector_model, "call_me", np.arange(3))
    register_gesture(gesture_classifier, clutch_corrector_model, "lets_drink", np.arange(3))

    # register_gesture("wave", np.arange(1))

    with open("./models/onedollar.pkl", "wb") as f:
        pickle.dump(gesture_classifier, f)


if __name__ == "__main__":
    main()
