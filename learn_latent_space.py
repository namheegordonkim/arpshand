import numpy as np
import os
import torch

from ml_models import SequenceAutoencoderClassifier
from utils import load_angles

STANDARD_LEN = 32


def resample(points):
    assert len(points) > 0
    filtered = [points[0]]
    prev = points[0]
    for pt in points[1:]:
        if np.linalg.norm(pt - prev) > 1e-3:
            filtered.append(pt)
            prev = pt
    points = filtered
    distFn = lambda p: np.linalg.norm(p[0] - p[1], ord=1)
    dists = list(map(distFn, zip(points, points[1:])))
    totalDist = sum(dists)
    distSoFar = 0
    currentSegment = 0
    ret = []
    for i in range(STANDARD_LEN - 1):
        targetDist = totalDist * (i / (STANDARD_LEN - 1))
        while (distSoFar + dists[currentSegment] < targetDist):
            distSoFar += dists[currentSegment]
            currentSegment += 1
        t = (targetDist - distSoFar) / dists[currentSegment]
        assert t >= 0 and t <= 1
        p0 = points[currentSegment]
        p1 = points[currentSegment + 1]
        ret.append(p0 + t * (p1 - p0))
    ret.append(points[-1])
    return np.array(ret)


def load_resampled_angles_with_labels(gesture_name):
    n_examples = len(os.listdir("./data/dynamic/{:s}/".format(gesture_name)))
    angles_matrix = []
    for i in range(n_examples):
        joint_angles, hand_angles, root_poses = load_angles(gesture_name, i)
        joint_angles_resampled = resample(joint_angles)
        angles_matrix.append(joint_angles_resampled.reshape(-1))
    angles_matrix = np.stack(angles_matrix)
    labels = np.repeat(gesture_name, n_examples)
    return angles_matrix, labels


def main():
    # load paper
    paper_features, paper_labels = load_resampled_angles_with_labels("paper")

    # load scissors
    scissors_features, scissors_labels = load_resampled_angles_with_labels("scissors")

    all_joint_angles = np.concatenate([paper_features, scissors_features], axis=0)
    all_labels = np.concatenate([paper_labels, scissors_labels])
    paper_yes = all_labels == "paper"
    scissors_yes = all_labels == "scissors"
    print(all_joint_angles.shape)

    n, d = all_joint_angles.shape
    k = len(np.unique(paper_yes))

    # encoder = PCA(n_components=2)
    # encoder = TSNE(n_components=2)
    # encoder = Isomap(n_components=2)
    encoder_classifier = SequenceAutoencoderClassifier(d, 256, k, [256, 256])
    encoder_classifier.fit(all_joint_angles, paper_yes.astype(np.float))

    torch.save(encoder_classifier, "./models/encoder_classifier.pkl")


if __name__ == "__main__":
    main()
