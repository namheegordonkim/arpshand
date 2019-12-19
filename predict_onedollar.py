import pickle
import numpy as np

from utils import load_gesture_lists

if __name__ == "__main__":
    joint_angles_list, _, _, _ = load_gesture_lists("paper", [10])
    with open("./models/onedollar.pkl", "rb") as f:
        model = pickle.load(f)
    joint_angles = np.concatenate(joint_angles_list)
    pred, score = model.predict_gesture(joint_angles)
    print(pred)
    print(score)

