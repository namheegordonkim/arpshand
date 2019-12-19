import numpy as np
import math


class OneDollarRecognizer:
    """Simple implementation of the 1$ Recognizer"""

    def __init__(self):
        # length of all paths after resampling
        self.standard_len = 128

        # array of (point array, string label) pairs
        self.known_gestures = []

    def define_gesture(self, points, label):
        self.known_gestures.append((self.standardize(points), label))

    def predict_gesture(self, points, window=False):
        assert len(self.known_gestures) > 0
        points = self.standardize(points)
        best_label = ""
        best_dist = np.inf
        for (path, label) in self.known_gestures:
            dist = self.path_dist(path, points, window)
            if dist < best_dist:
                best_dist = dist
                best_label = label
        # score = best_dist
        return best_label, best_dist

    def path_dist(self, x1, x2, window):
        n, d = x1.shape
        disp = x1 - x2
        D = np.zeros(n)
        for i in range(n):
            D[i] = np.sum(np.abs(disp[i])**(1./2)) ** (2)
        err = np.sum(D)
        # np.sum(disp ** 2) ** 1/2
        # distfn = lambda v: np.linalg.norm(v, ord=2)
        # err = np.apply_along_axis(distfn, 1, disp)
        if window:
            err = err * self.get_window(len(err))
        return np.sum(err) / self.standard_len

    def get_window(self, n):
        weights = []
        f = n // 4
        for i in range(f):
            weights.append(i / f)
        for i in range(n - 2 * f):
            weights.append(1)
        for i in range(f):
            weights.append(1 - i / f)
        return np.asarray(weights)

    def resample(self, points):
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
        for i in range(self.standard_len - 1):
            targetDist = totalDist * (i / (self.standard_len - 1))
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

    # def scale(self, points):
    #     assert len(points.shape) == 2
    #     n = points.shape[1]
    #     min = np.full(n, np.inf)
    #     max = np.full(n, -np.inf)
    #     for x in points:
    #         assert len(x) == len(min)
    #         assert len(x) == len(max)
    #         min = np.minimum(x, min)
    #         max = np.maximum(x, max)
    #     # TODO: prevent division by zero or by extremely small values
    #     fn = lambda v: (v - min) / (max - min)
    #     return np.apply_along_axis(fn, 1, points)

    def standardize(self, points):
        # NOTE: no rotation because it doesn't make sense with our
        # high number of dimensions
        # NOTE: scaling is disabled because it removes useful information
        return self.resample(np.asarray(points))
        # return self.scale(self.resample(np.array(points)))
