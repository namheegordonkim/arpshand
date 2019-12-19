import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler


def main():
    with open("./data/dynamics_disps.pkl", "rb") as f:
        data_dict = pickle.load(f)

    current_states = data_dict["current_states"]
    next_states = data_dict["next_states"]
    labels = data_dict["labels"]

    scaler = StandardScaler()
    compressor = PCA(n_components=2)

    current_states_reshaped = np.stack([s.reshape(-1) for s in current_states])
    next_states_reshaped = np.stack([s.reshape(-1) for s in next_states])

    X_current = scaler.fit_transform(current_states_reshaped)
    X_next = scaler.transform(next_states_reshaped)
    compressor.fit(X_current)

    alpha = 1e-1
    gamma = 1e-2
    all_gesture_dynamics_model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    # learn all gesutres
    all_gesture_dynamics_model.fit(X_current, X_next)

    dynamics_dict = {"all": all_gesture_dynamics_model}

    gesture_names = ["ok", "thumbs_up", "paper", "scissors", "lets_drink", "call_me"]
    for gesture_name in gesture_names:
        print(gesture_name)
        gesture_yes = labels == gesture_name
        dynamics_model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        dynamics_model.fit(X_current[gesture_yes], X_next[gesture_yes])
        dynamics_dict[gesture_name] = dynamics_model

    Xhat_next = all_gesture_dynamics_model.predict(X_current)
    training_err = np.mean((X_next - Xhat_next) ** 2)
    print("Training error (all gestures): {:f}".format(training_err))

    with open("./models/dynamics_models.pkl", "wb") as f:
        pickle.dump(dynamics_dict, f)

    with open("./models/dynamics_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("./models/dynamics_compressor.pkl", "wb") as f:
        pickle.dump(compressor, f)


if __name__ == "__main__":
    main()
