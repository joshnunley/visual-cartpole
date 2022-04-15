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

    def __init__(
        self,
        worlds_init,
        kernels,
        kernel_weights,
        growth_params,
        num_channels,
        time_step,
    ):
        self.worlds = worlds_init
        self.kernels = kernels
        self.kernel_weights = kernel_weights
        self.growth_params = growth_params
        self.time_step = time_step
        self.num_channels = num_channels

        # TODO: There are lots of consistency checks I should run here

    def update(self):
        world_shape = np.shape(self.worlds[0])
        for channel_list_ind in range(self.num_channels):
            kernel_channel_list = self.kernels[channel_list_ind]
            if not kernel_channel_list:
                continue
            kernel_channel_list_len = len(kernel_channel_list)
            convolved_worlds_ind = 0
            convolved_worlds = np.zeros(shape=(kernel_channel_list_len, *world_shape))
            for group_ind in range(kernel_channel_list_len):
                kernel_group = kernel_channel_list[group_ind]
                kernel_weights_group = self.kernel_weights[channel_list_ind][group_ind]
                growth_params_group = self.growth_params[channel_list_ind][group_ind]

                convolve_channels = list(kernel_group.keys())
                growth_func = lambda x: self._growth(x, *tuple(growth_params_group))

                convolved_worlds[convolved_worlds_ind, ...] = growth_func(
                    np.sum(
                        [
                            kernel_weights_group[convolve_channel]
                            * signal.convolve(
                                self.worlds[convolve_channel],
                                kernel_group[convolve_channel],
                                mode="same",
                            )
                            for convolve_channel in convolve_channels
                        ],
                        axis=0,
                    )
                )

                convolved_worlds_ind += 1

            self.worlds[channel_list_ind, ...] = self._clip(
                self.worlds[channel_list_ind, ...]
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

        for channel_list_ind in range(self.num_channels):
            kernel_channel_list = self.kernels[channel_list_ind]
            for group_ind in range(len(kernel_channel_list)):
                kernel_group = kernel_channel_list[group_ind]
                kernel_weights_group = self.kernel_weights[channel_list_ind][group_ind]
                for channel in kernel_group.keys():
                    kernel_shape = np.shape(kernel_group[channel])
                    kernel_size = np.prod(kernel_shape)
                    kernel_group[channel] = np.reshape(
                        2 * (param_array[params_ind : params_ind + kernel_size] - 0.5),
                        (kernel_shape),
                    )
                    params_ind += kernel_size

                    kernel_weights_group[channel] = 10 * (param_array[params_ind] - 0.5)
                    params_ind += 1

                growth_params_shape = np.shape(
                    self.growth_params[channel_list_ind][group_ind]
                )
                growth_params_size = np.prod(growth_params_shape)
                self.growth_params[channel_list_ind][group_ind] = (
                    param_array[params_ind : params_ind + growth_params_size] + 0.0001
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

        for channel_kernel_list in self.kernels:
            for kernel_group in channel_kernel_list:
                for channel in kernel_group.keys():
                    kernel_size = np.size(kernel_group[channel])
                    growth_param_size = 4
                    num_weights = 1

                    num_params += kernel_size + growth_param_size + num_weights

        # for time step
        num_params += 1

        return num_params

    def _growth(self, input, scaling, mean_bias, std, interpolation):
        mean_bias = 20 * (mean_bias - 0.5)
        sigmoid = 2 / (1 + np.exp((-input + mean_bias) / std)) - 1
        bell = 2 * np.exp(-(((input - mean_bias) / std) ** 2) / 2) - 1

        return scaling * ((1 - interpolation) * sigmoid + interpolation * bell)

    def _clip(self, x):
        return np.clip(x, 0, 1)


def LeniaConstructor(
    num_channels, time_step, kernel_radius, world_shape, kernel_architecture
):
    worlds_init = np.random.uniform(size=(num_channels, *world_shape))

    kernel_size = 2 * kernel_radius + 1
    kernel_shape = tuple([kernel_size for x in world_shape])
    kernels = []
    kernel_weights = []
    growth_params = []
    for channel_kernel_list in kernel_architecture:
        channel_kernel_groups = []
        channel_kernel_weights_groups = []
        channel_growth_param_groups = []
        for convolve_channel_group in channel_kernel_list:
            channel_kernel_group = {}
            channel_kernel_weights_group = {}
            for channel in convolve_channel_group:
                channel_kernel_group[channel] = np.random.uniform(size=kernel_shape)
                channel_kernel_weights_group[channel] = np.random.uniform()
            channel_kernel_groups.append(channel_kernel_group)
            channel_kernel_weights_groups.append(channel_kernel_weights_group)
            channel_growth_param_groups.append(np.random.uniform(size=(4)))
        kernels.append(channel_kernel_groups)
        kernel_weights.append(channel_kernel_weights_groups)
        growth_params.append(channel_growth_param_groups)

    return Lenia(
        worlds_init, kernels, kernel_weights, growth_params, num_channels, time_step
    )

