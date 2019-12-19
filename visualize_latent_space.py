import torch
import matplotlib.pyplot as plt
import numpy as np
from learn_latent_space import load_resampled_angles_with_labels
from ml_models import SequenceAutoencoderClassifier, MultiLayerPerceptron


def main():
    # load paper
    paper_features, paper_labels = load_resampled_angles_with_labels("paper")
    # load scissors
    scissors_features, scissors_labels = load_resampled_angles_with_labels("scissors")

    all_joint_angles = np.concatenate([paper_features, scissors_features], axis=0)
    all_labels = np.concatenate([paper_labels, scissors_labels])
    paper_yes = all_labels == "paper"
    scissors_yes = all_labels == "scissors"

    encoder_classifier: SequenceAutoencoderClassifier = torch.load("./models/encoder_classifier.pkl",
                                                                   map_location="cpu")
    Z = encoder_classifier.transform(all_joint_angles)

    plt.figure()
    plt.scatter(Z[paper_yes, 0], Z[paper_yes, 1], label="paper")
    plt.scatter(Z[scissors_yes, 0], Z[scissors_yes, 1], label="scissors")
    plt.legend()
    plt.show()

    all_joint_angles_estimated = encoder_classifier.inverse_transform(Z)
    all_joint_angles_reshaped = np.concatenate([a.reshape(32, 15) for a in all_joint_angles], axis=0)
    all_joint_angles_estimated_reshaped = np.concatenate([a.reshape(32, 15) for a in all_joint_angles_estimated], axis=0)

    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)
    for j in range(15):
        ax1.plot(all_joint_angles_reshaped[:, j])
        ax2.plot(all_joint_angles_estimated_reshaped[:, j])
    plt.show()

    print(all_joint_angles_estimated)

if __name__ == "__main__":
    main()
