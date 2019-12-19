import pickle

import matplotlib.pyplot as plt
import numpy as np


def generate_trajectory(initial_state, dynamics_model, n_timesteps, speed):
    n_features = len(initial_state.squeeze())
    X_generated = np.zeros([n_timesteps, n_features])
    s = np.copy(initial_state)
    for i in range(n_timesteps):
        X_generated[i, :] = s
        next = dynamics_model.predict(s)
        delta = next - s
        s = s + delta * speed
    return X_generated


def main():
    np.random.seed(1)

    with open("./models/dynamics_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("./models/dynamics_compressor.pkl", "rb") as f:
        compressor = pickle.load(f)
    with open("./models/dynamics_models.pkl", "rb") as f:
        dynamics_dict = pickle.load(f)
    with open("./data/dynamics_disps.pkl", "rb") as f:
        data_dict = pickle.load(f)

    current_states = data_dict["current_states"]
    next_states = data_dict["next_states"]
    labels = data_dict["labels"]
    first_yes = data_dict["first_yes"]
    current_states_reshaped = np.stack([s.reshape(-1) for s in current_states])
    next_states_reshaped = np.stack([s.reshape(-1) for s in next_states])
    _, n_features = current_states_reshaped.shape

    X_current = scaler.fit_transform(current_states_reshaped)
    X_next = scaler.transform(next_states_reshaped)

    n_timesteps = 30

    plt.figure()
    gesture_names = ["call_me"]

    for gesture_idx, gesture_name in enumerate(gesture_names):
        gesture_yes = labels == gesture_name
        dynamics_model = dynamics_dict[gesture_name]
        X_generated_list = []
        X_first = X_current[gesture_yes & first_yes]
        idx = np.random.choice(X_first.shape[0], 1)

        speeds = [0.5, 1.0, 1.5]
        for i, speed in enumerate(speeds):
            s = X_first[idx]
            s += np.random.normal(0, 1e-1, s.shape)
            X_generated = generate_trajectory(s, dynamics_model, n_timesteps, speed)
            X_generated_list.append(X_generated)

            Z_generated = compressor.transform(X_generated)
            plt.scatter(Z_generated[:, 0], Z_generated[:, 1], alpha=0.5, color="C{:d}".format(i),
                        label="Speed={:.2f}".format(speed))
            for current, next in zip(Z_generated, Z_generated[1:]):
                delta = next - current
                plt.arrow(current[0], current[1], delta[0], delta[1], color="C{:d}".format(i), alpha=0.5, width=0.05,
                          length_includes_head=True)

            with open("./data/{:s}_generated_{:.2f}.pkl".format(gesture_name, speed), "wb") as f:
                pickle.dump(scaler.inverse_transform(X_generated), f)

        plt.title("Synthetic trajectories of gesture \"{:s}\"".format(gesture_name))
        plt.legend()
        plt.show()

        fig, axes = plt.subplots(nrows=len(speeds), ncols=1)
        for X_generated, ax, speed in zip(X_generated_list, axes, speeds):
            _, n_features = X_generated.shape
            for j in range(n_features):
                ax.plot(X_generated[:, j])
            ax.set_title("Speed={:.2f}".format(speed))
        plt.show()


if __name__ == "__main__":
    main()
