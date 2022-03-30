from lenia import Lenia

import numpy as np
from collections import defaultdict

import matplotlib

matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from matplotlib import animation

world_size = 64
radius = 5
time_step = 0.05
growth_mean = 0.135
growth_std = 0.015

kernel_size = 2 * radius + 1

worlds = [
    np.random.uniform(size=(world_size, world_size)),
    np.random.uniform(size=(world_size, world_size)),
]
kernels = np.ones(shape=(1, kernel_size, kernel_size))
kernels[0, radius, radius] = 0
kernels[0, ...] = kernels[0, ...] / np.sum(kernels[0, ...])


kernels_dict = {(0, 0): kernels, (0, 1): kernels, (1, 1): kernels}

# Use all the previously defined stuff to create a 2 dimensional
# instance of Lenia
num_channels = 2
growth_params = {
    (0, 0): np.array([[2, 1, growth_mean, growth_std]]),
    (1, 1): np.array([[2, 1, growth_mean, growth_std]]),
    (0, 1): np.array([[1, 0.5, growth_mean, growth_std]]),
}
lenia2d = Lenia(worlds, kernels_dict, growth_params, num_channels, time_step)


def my_func(i):
    print(i)
    lenia2d.update()
    plt.imshow(
        lenia2d.worlds[0],
        vmin=0,
        vmax=1,
    )


fig = plt.figure()
anim = animation.FuncAnimation(
    fig=fig, func=my_func, frames=100, interval=50, blit=False
)

plt.show()
# anim.save("lenia.gif", dpi=300, writer=animation.PillowWriter(fps=25))
