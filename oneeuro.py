import numpy as np


class OneEuroFilter:

    def __init__(self, rate, min_cutoff, beta, d_cutoff):
        self.rate = rate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.cutoff = min_cutoff
        self.first_time = True
        self.hat_x_prev = 0
        self.hat_x = 0
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def get_alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        t_e = 1.0 / self.rate
        return 1.0 / (1.0 + tau / t_e)

    def filter(self, x):
        if self.first_time:
            self.first_time = False
            dx = 0
        else:
            dx = (x - self.x_filter.hat_x_prev) * self.rate

        alpha = self.get_alpha(self.d_cutoff)
        e_dx = self.dx_filter.filter(dx, alpha)

        self.cutoff = self.min_cutoff + self.beta * np.abs(e_dx)

        alpha = self.get_alpha(self.cutoff)
        e_x = self.x_filter.filter(x, alpha)
        return e_x


class LowPassFilter:
    def __init__(self):
        self.first_time = True
        self.hat_x_prev = 0
        self.hat_x = 0

    def filter(self, x, alpha):
        if self.first_time:
            self.first_time = False
            self.hat_x_prev = x

        self.hat_x = alpha * x + (1 - alpha) * self.hat_x_prev
        self.hat_x_prev = self.hat_x
        return self.hat_x
