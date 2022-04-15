from multiprocessing.sharedctypes import Value
from scipy import signal
import numpy as np


class Lenia:
    """
    worlds_init is a numpy array of starting conditions the same number of dimensions as num_channels.
    These starting conditions may be square arrays of any dimension consisting of values
    between 0 and 1, but they must all have the same dimensions and sizes of dimensions.

    kernels is a dictionary mapping tuples of channels to a numpy array of kernels. Kernels are arrays of the same
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
        self.kernels = kernels
        self.growth_params = growth_params
        self.time_step = time_step
        self.num_channels = num_channels

        # TODO: There are lots of consistency checks I should run here

    def update(self):
        for channel_ind in range(self.num_channels):
            kernels_acting_on_channel = [
                key for key in self.kernels.keys() if key[0] == channel_ind
            ]
            num_convolved_worlds = sum(
                [np.shape(self.kernels[key])[0] for key in kernels_acting_on_channel]
            )
            world_shape = np.shape(self.worlds[0])
            convolved_worlds = np.zeros(shape=(num_convolved_worlds, *world_shape))

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
                    convolved_worlds_ind += 1

            self.worlds[channel_ind, ...] = self._clip(
                self.worlds[channel_ind]
                + self.time_step * np.sum(convolved_worlds, axis=0)
            )

    def set_params(self, param_array, included_worlds=[]):
        num_world_params = self.worlds[included_worlds, ...].size

        params_ind = 0

        self.worlds[included_worlds, ...] = np.reshape(
            param_array[params_ind : params_ind + num_world_params],
            (len(included_worlds), *np.shape(self.worlds)[1:]),
        )

        params_ind += num_world_params

        for channel_tuple in self.kernels.keys():
            kernels_shape = np.shape(self.kernels[channel_tuple])
            kernels_size = np.prod(kernels_shape)
            self.kernels[channel_tuple] = np.reshape(
                param_array[params_ind : params_ind + kernels_size], (kernels_shape)
            )
            params_ind += kernels_size

            growth_params_shape = np.shape(self.growth_params[channel_tuple])
            growth_params_size = np.prod(growth_params_shape)
            self.growth_params[channel_tuple] = np.reshape(
                param_array[params_ind : params_ind + growth_params_size],
                (growth_params_shape),
            )
            params_ind += growth_params_size

        self.time_step = param_array[params_ind]

    def get_num_params(self, num_included_worlds=0):
        if num_included_worlds > self.num_channels:
            raise ValueError(
                "num_included_worlds is greater than the number of channels"
            )
        num_params = 0

        world_shape = np.shape(self.worlds)
        num_world_params = np.prod(world_shape[1:]) * (num_included_worlds)
        num_params += num_world_params

        for channel_tuple in self.kernels.keys():
            kernels_shape = np.shape(self.kernels[channel_tuple])
            kernels_size = np.prod(kernels_shape)

            growth_params_shape = np.shape(self.growth_params[channel_tuple])
            growth_params_size = np.prod(growth_params_shape)

            num_params += kernels_size + growth_params_size

        num_params += 1

        return num_params

    def _growth(self, x, a, b, mean, std):
        return a * np.exp(-(((x - mean) / std) ** 2) / 2) - b

    def _clip(self, x):
        return np.clip(x, 0, 1)


def LeniaConstructor(
    num_channels, time_step, kernel_radius, world_shape, kernel_architecture
):
    worlds_init = np.random.uniform(size=(num_channels, *world_shape))

    kernel_size = 2 * kernel_radius + 1
    kernel_shape = tuple([kernel_size for x in world_shape])
    kernels = {}
    growth_params = {}
    for channel_tuple in kernel_architecture.keys():
        num_kernels = kernel_architecture[channel_tuple]
        channel_tuple_kernels = np.random.uniform(size=(num_kernels, *kernel_shape))
        channel_tuple_growth_params = np.random.uniform(size=(num_kernels, 4))
        kernels[channel_tuple] = channel_tuple_kernels
        growth_params[channel_tuple] = channel_tuple_growth_params

    return Lenia(worlds_init, kernels, growth_params, num_channels, time_step)

