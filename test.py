import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import animation

import lenia_modified
import neural_lenia
import lenia

world_shape = (64, 64)

modified_constructor_params = {
    "kernel_radius": 5,
    "time_step": 0.05,
    "num_channels": 1,
    "world_shape": world_shape,
    "kernel_architecture": {(0, 0): 1},
}
neural_constructor_params = {
    "kernel_radius": 5,
    "time_step": 0.05,
    "num_channels": 1,
    "world_shape": world_shape,
    "kernel_architecture": [[[0]]],
}

lenia_modified = lenia_modified.LeniaConstructor(**modified_constructor_params)

lenia = lenia.LeniaConstructor(**modified_constructor_params)
lenia = lenia_modified

lenia.kernels[(0, 0)][:] = 1
lenia.kernels[(0, 0)][0, 5, 5] = 0
lenia.kernels[(0, 0)] /= np.sum(lenia.kernels[(0, 0)])

lenia.growth_params[(0, 0)][0][0] = 1
lenia.growth_params[(0, 0)][0][1] = (0.135 + 1) / 2
lenia.growth_params[(0, 0)][0][2] = 0.015
lenia.growth_params[(0, 0)][0][3] = 1

# x = np.arange(-1, 1, 0.01)
# plt.plot(x, lenia._growth(x, *lenia.growth_params[(0, 0)][0]))
# plt.show()

neural_lenia1 = neural_lenia.LeniaConstructor(**neural_constructor_params)

neural_lenia1.kernels[0][0][0][:] = 1
neural_lenia1.kernels[0][0][0][5, 5] = 0
neural_lenia1.kernels[0][0][0] /= np.sum(neural_lenia1.kernels[0][0][0][:])

neural_lenia1.growth_params[0][0][0] = 1
neural_lenia1.growth_params[0][0][1] = (0.135 + 1) / 2
neural_lenia1.growth_params[0][0][2] = 0.015
neural_lenia1.growth_params[0][0][3] = 1

lenia = neural_lenia1
lenia.update()

fig = plt.figure()
im = plt.imshow(lenia.worlds[0], cmap="gray", vmin=0, vmax=1,)


def my_func(i):
    print(i)

    im.set_array(lenia.worlds[0])

    plt.title("Frame: " + str(i))

    lenia.update()

    return [im]


anim = animation.FuncAnimation(
    fig=fig, func=my_func, frames=100, interval=10, blit=False
)
anim.save("test.gif", dpi=300, writer=animation.PillowWriter(fps=10))
