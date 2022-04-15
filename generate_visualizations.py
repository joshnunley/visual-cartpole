from neural_lenia import Lenia, LeniaConstructor
from fitness_functions import MSE, correlation_fitness
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

from CartPoleEnv import CartPoleEnv

import pickle


with open("best_individual10", "rb") as f:
    best_individual = pickle.load(f)

########################
# Cart Pole
########################

cart_pole = CartPoleEnv()
cart_pole.reset()

########################
# Construct Model
########################

lenia = LeniaConstructor(**best_individual["constructor_params"])

lenia.set_params(
    best_individual["lenia_params"], included_worlds=best_individual["included_worlds"]
)


########################
# Test Model
########################


"""
total_fitness = 0
num_time_steps = 15
all_worlds = np.zeros(shape=(*np.shape(lenia.worlds[0]), num_time_steps))
for i in range(num_time_steps):
    lenia.update()
    all_worlds[..., i] = lenia.worlds[0]

total_fitness = correlation_fitness(np.copy(bold_image[..., 1 : num_time_steps + 1]), np.copy(all_worlds))

print(total_fitness)
"""
########################
# Generate Plots
########################
print(lenia.growth_params)

# plt.imshow(lenia.kernels[(1, 0)][0], cmap="gray")
# plt.show()
# plt.imshow(lenia.kernels[(1, 0)][1], cmap="gray")
# plt.show()
# plt.imshow(lenia.kernels[(1, 1)][0], cmap="gray")
# plt.show()
# plt.imshow(lenia.kernels[(1, 1)][1], cmap="gray")
# plt.show()


force_scaling = 2 * best_individual["lenia_params"][-3] - 1
print(force_scaling, lenia.time_step)
print(lenia.kernel_weights)

# Assume maximum value in data is 1 and minimum value is 0
data_scaling = best_individual["lenia_params"][-2] + 0.01
data_bias = np.min([1 - data_scaling, best_individual["lenia_params"][-1]])


def map_state_to_f(world):
    mid_point = int((np.shape(world)[1] - 1) / 2)
    left_world = world[:, 0:mid_point]
    right_world = world[:, mid_point + 1 :]

    right_force = np.sum(left_world > 0.5)
    left_force = np.sum(right_world > 0.5)
    print(left_force, right_force)

    # make sure sign is right here
    return right_force - left_force


fig = plt.figure()
im1 = plt.imshow(lenia.worlds[0], cmap="gray", vmin=0, vmax=1,)
im2 = plt.imshow(
    lenia.worlds[1], cmap="Blues", alpha=0.3 * lenia.worlds[1], vmin=0, vmax=1,
)
im3 = plt.imshow(
    lenia.worlds[2], cmap="Reds", alpha=0.3 * lenia.worlds[2], vmin=0, vmax=1,
)


def my_func(i):
    print(i)

    vis = cart_pole.get_visual_array()
    lenia.worlds[0] = data_scaling * vis + data_bias

    im1.set_array(lenia.worlds[0])
    im2.set_array(lenia.worlds[1])
    im3.set_array(lenia.worlds[2])

    for j in range(best_individual["num_updates"]):
        lenia.update()

    f = map_state_to_f(lenia.worlds[2])
    print("force: ", force_scaling * f)
    plt.title("Frame: " + str(i) + " Force: " + str(force_scaling * f))
    cart_pole_state, reward, done, d = cart_pole.step(force_scaling * f)

    return [im1, im2, im3]


anim = animation.FuncAnimation(
    fig=fig, func=my_func, frames=200, interval=10, blit=False
)
anim.save("cart_pole11.gif", dpi=300, writer=animation.PillowWriter(fps=10))

