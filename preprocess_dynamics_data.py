import pickle

import numpy as np

from utils import load_all_coords, load_all_quats, load_all_disps


def main():
    # ok_joint_quats, ok_hand_quats, ok_root_poses_list = load_all_root_poses("ok")
    # tu_joint_quats, tu_hand_quats, tu_root_poses_list = load_all_root_poses("thumbs_up")

    all_current = []
    all_next = []
    all_labels = []
    all_first_yes = []

    gesture_names = ["ok", "thumbs_up", "paper", "scissors", "lets_drink", "call_me"]
    for gesture_name in gesture_names:
        # features_list, _, _ = load_all_quats(gesture_name)
        features_list = load_all_disps(gesture_name)
        #
        # ok_joint_quats_list, _, _ = load_all_quats("ok")
        # tu_joint_quats_list, _, _ = load_all_quats("thumbs_up")
        # pa_joint_quats_list, _, _ = load_all_quats("paper")

        # for each root_poses sequence, we want to extract "current frame" and "next frame"
        # i.e. remove the last frame from video and call it current frame,
        # and shift video by one frame and call it next frame

        # TODO: populate the matrices
        current = []
        next = []
        labels = []
        first_yes = []
        for features in features_list:
            n_frames = features.shape[0]
            current_ = features[:-1, :]
            next_ = features[1:, :]
            current.append(current_)
            next.append(next_)
            first_yes_ = np.zeros(n_frames-1, dtype=np.bool)
            first_yes_[0] = True
            first_yes.append(first_yes_)
            for _ in range(n_frames-1):
                labels.append(gesture_name)
        current = np.concatenate(current, axis=0)
        next = np.concatenate(next, axis=0)
        labels = np.asarray(labels)
        first_yes = np.concatenate(first_yes)

        all_current.append(current)
        all_next.append(next)
        all_labels.append(labels)
        all_first_yes.append(first_yes)

    all_current = np.concatenate(all_current, axis=0)
    all_next = np.concatenate(all_next, axis=0)
    all_labels = np.concatenate(all_labels)
    all_first_yes = np.concatenate(all_first_yes)

    data_dict = {
        "current_states": all_current,
        "next_states": all_next,
        "labels": all_labels,
        "first_yes": all_first_yes
    }

    with open("./data/dynamics_disps.pkl", "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    main()
