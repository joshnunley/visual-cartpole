from scipy import signal
import numpy as np
from collections import defaultdict


class Lenia:
    """
    worlds_init is a list of starting conditions the same length as num_channels.
    These starting conditions may be square arrays of any dimension consisting of values
    between 0 and 1, but they must all have the same dimensions and sizes of dimensions.

    kernels is a default dictionary mapping tuples of channels to a numpy array of kernels. Kernels are arrays of the same
    dimension as worlds_init, but the size of each dimension must be less than that of worlds_init.
    For example, (1, 2) is a possible key, the value of which is the kernel that convolves with channel 2
    and acts on channel 1.

    growths is a dictionary mapping tuples of channels to a 2d numpy array of parameters for a function
    of the form a * bell(x; mean, std) - b where a and b are greater than or equal to 0. The parameters
    should be in the order [a, b, mean, std]. These should align with the kernel dictionary.

    time_step is a floating point value between 0 and 1, but usually closer to 0. This determines
    how much to "integrate" the values at each step of the simulation.

    num_channels is the integer representing the number of channels in the simulation.

    """

    def __init__(self, worlds_init, kernels, growth_params, num_channels, time_step):
        self.worlds = worlds_init
        self.world_shape = np.shape(self.worlds[0])
        self.kernels = kernels
        self.growth_params = growth_params
        self.time_step = time_step
        self.num_channels = num_channels

    def update(self):
        for channel_ind in range(self.num_channels):
            kernels_acting_on_channel = [
                key for key in self.kernels.keys() if key[0] == channel_ind
            ]
            num_convolved_worlds = sum(
                [np.shape(self.kernels[key])[0] for key in kernels_acting_on_channel]
            )
            convolved_worlds = np.zeros(shape=(num_convolved_worlds, *self.world_shape))

            convolved_worlds_ind = 0
            for kernel_key in kernels_acting_on_channel:
                for kernel_ind in range(np.shape(self.kernels[kernel_key])[0]):
                    kernel = self.kernels[kernel_key][kernel_ind, ...]
                    growth_func = lambda x: self._growth(
                        x, *tuple(self.growth_params[kernel_key][kernel_ind, :])
                    )
                    convolve_channel = kernel_key[1]
                    convolved_worlds[convolved_worlds_ind, ...] = growth_func(
                        signal.convolve(
                            self.worlds[convolve_channel], kernel, mode="same"
                        )
                    )
                    if kernel_key == (0, 1):
                        print(np.sum(convolved_worlds[convolved_worlds_ind]))
                    convolved_worlds_ind += 1

            self.worlds[channel_ind] = self._clip(
                self.worlds[channel_ind]
                + self.time_step * np.sum(convolved_worlds, axis=0)
            )

    def _growth(self, x, a, b, mean, std):
        return a * np.exp(-(((x - mean) / std) ** 2) / 2) - b

    def _clip(self, x):
        return np.clip(x, 0, 1)
