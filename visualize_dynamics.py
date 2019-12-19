import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_dynamics(dynamics_model, compressor, Ztest, color):
    Xtest = compressor.inverse_transform(Ztest)
    Xpred = dynamics_model.predict(Xtest)
    Zpred = compressor.transform(Xpred)
    for current, next in zip(Ztest, Zpred):
        delta = next - current
        delta /= np.linalg.norm(delta)
        plt.arrow(current[0], current[1], delta[0], delta[1], color=color, alpha=0.5, width=0.03,
                  length_includes_head=True)


def main():
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
    ok_yes = labels == "ok"
    tu_yes = labels == "thumbs_up"
    first_yes = data_dict["first_yes"]

    current_states_reshaped = np.stack([s.reshape(-1) for s in current_states])
    next_states_reshaped = np.stack([s.reshape(-1) for s in next_states])

    # X_current = current_states_reshaped
    # X_next = next_states_reshaped
    X_current = scaler.fit_transform(current_states_reshaped)
    X_next = scaler.transform(next_states_reshaped)

    Z_current = compressor.fit_transform(X_current)
    Z_next = compressor.transform(X_next)

    all_gesture_dynamics_model = dynamics_dict["all"]
    Xhat_next = all_gesture_dynamics_model.predict(X_current)
    Zhat_next = compressor.transform(Xhat_next)

    Ztest_current = np.dstack(np.meshgrid(np.linspace(-15, 15, num=50), np.linspace(-15, 15, num=50))).reshape(-1, 2)

    gesture_names = ["ok", "thumbs_up", "paper", "scissors", "lets_drink", "call_me"]

    # plt.figure()
    # for gesture_name in gesture_names:
    #     dynamics_model = dynamics_dict[gesture_name]
    #     gesture_yes = labels == gesture_name
    #     plt.scatter(Z_current[gesture_yes, 0], Z_current[gesture_yes, 1], label=gesture_name, alpha=0.5)
        # plt.scatter(Z_current[first_yes, 0], Z_current[first_yes, 1], alpha=1, s=10 ** 2)

        # draw lines
        # for current, next in zip(Z_current[gesture_yes], Z_next[gesture_yes]):
        #     delta = next - current
        #     plt.arrow(current[0], current[1], delta[0], delta[1], color="blue", alpha=0.2, width=0.1,
        #               length_includes_head=True)
    # plt.legend()
    # plt.show()
    #
    # plt.figure()

    for i, gesture_name in enumerate(gesture_names):
        plt.figure()
        for gesture_name_ in gesture_names:
            gesture_yes = labels == gesture_name_
            plt.scatter(Z_current[gesture_yes, 0], Z_current[gesture_yes, 1], label=gesture_name_, alpha=0.5)

        gesture_yes = labels == gesture_name
        dynamics_model = dynamics_dict[gesture_name]
        plot_dynamics(dynamics_model, compressor, Ztest_current, "C{:d}".format(i))
        plt.title("Dynamics of gesture \"{:s}\"".format(gesture_name))
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.legend()
        # plt.show()
        plt.savefig("./gesture_{:s}.pdf".format(gesture_name))
        plt.close()

    # plt.figure()
    # plt.scatter(Z_current[ok_yes, 0], Z_current[ok_yes, 1], label="ok", alpha=0.5)
    # plt.scatter(Z_current[tu_yes, 0], Z_current[tu_yes, 1], label="thumbs_up", alpha=0.5)
    # plot_dynamics(all_gesture_dynamics_model, compressor, Ztest_current, "red")
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.scatter(Z_current[ok_yes, 0], Z_current[ok_yes, 1], label="ok", alpha=0.5)
    # plt.scatter(Z_current[tu_yes, 0], Z_current[tu_yes, 1], label="thumbs_up", alpha=0.5)
    # plot_dynamics(ok_dynamics_model, compressor, Ztest_current, "blue")
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.scatter(Z_current[ok_yes, 0], Z_current[ok_yes, 1], label="ok", alpha=0.5)
    # plt.scatter(Z_current[tu_yes, 0], Z_current[tu_yes, 1], label="thumbs_up", alpha=0.5)
    # plot_dynamics(tu_dynamics_model, compressor, Ztest_current, "orange")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
