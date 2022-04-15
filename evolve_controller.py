from xml.etree.ElementInclude import include
from neural_lenia import Lenia, LeniaConstructor
from fitness_functions import MSE, correlation_fitness
from EvolSearch import EvolSearch

import numpy as np

from CartPoleEnv import CartPoleEnv

import pickle

import matplotlib.pyplot as plt

use_best_individual = False
with open("best_individual7", "rb") as f:
    best_individual = pickle.load(f)

########################
# Fitness Function
########################
def map_state_to_f(world):
    mid_point = int((np.shape(world)[1] - 1) / 2)
    left_world = world[:, 0:mid_point]
    right_world = world[:, mid_point + 1 :]

    right_force = np.sum(left_world > 0.5)
    left_force = np.sum(right_world > 0.5)

    return right_force - left_force


def lenia_fitness(lenia_parameters, lenia, cart_pole, included_worlds, num_updates):
    force_scaling = 2 * lenia_parameters[-3] - 1

    # Assume maximum value in data is 1 and minimum value is 0
    data_scaling = lenia_parameters[-2] + 0.01
    data_bias = np.min([1 - data_scaling, lenia_parameters[-1]])

    lenia.set_params(lenia_parameters, included_worlds=included_worlds)

    num_time_steps = 300
    total_fitness = 0
    num_trials = 6  # should be even
    cart_pole.seed(0)
    for k in range(num_trials):
        if k < int(num_trials / 2):
            cart_pole.reset(-1)
        else:
            cart_pole.reset(1)
        for i in range(num_time_steps):
            input = data_scaling * cart_pole.get_visual_array() + data_bias
            lenia.worlds[0] = input
            for j in range(num_updates):
                lenia.update()

            f = map_state_to_f(lenia.worlds[1])
            cart_pole_state, reward, done, d = cart_pole.step(force_scaling * f)
            total_fitness += reward

            if done:
                break
    return total_fitness / num_trials


########################
# Construct Lenia
########################

cart_pole = CartPoleEnv()

constructor_params = {
    "kernel_radius": 1,
    "time_step": 0.05,
    "num_channels": 2,
    "world_shape": np.shape(cart_pole.get_visual_array()),
    # "kernel_architecture": {(1, 0): 2, (1, 1): 2},
    "kernel_architecture": [[], [[0], [1], [0], [1]]],
}

if use_best_individual:
    lenia = LeniaConstructor(**best_individual["constructor_params"])
else:
    lenia = LeniaConstructor(**constructor_params)

########################
# Evolve Solutions
########################
if use_best_individual:
    included_worlds = best_individual["included_worlds"]
    num_updates = best_individual["num_updates"]
else:
    included_worlds = [1]
    num_updates = 5
genotype_size = lenia.get_num_params(num_included_worlds=len(included_worlds)) + 3
pop_size = 100

fitness_function = lambda params: lenia_fitness(
    params, lenia, cart_pole, included_worlds, num_updates
)
evol_params = {
    "num_processes": 6,
    "pop_size": pop_size,  # population size
    "genotype_size": genotype_size,  # dimensionality of solution
    "fitness_function": fitness_function,  # custom function defined to evaluate fitness of a solution
    "elitist_fraction": 0.1,  # fraction of population retained as is between generations
    "mutation_variance": 0.01,  # mutation noise added to offspring.
}

initial_pop = np.random.uniform(size=(pop_size, genotype_size))

mean = 0
for i in range(pop_size):
    mean += fitness_function(initial_pop[i])
    print(mean / (i + 1))

exit()

if use_best_individual:
    initial_pop[0] = best_individual["lenia_params"]

evolution = EvolSearch(evol_params, initial_pop)

save_best_individual = {
    "constructor_params": constructor_params,
    "included_worlds": included_worlds,
    "num_updates": num_updates,
    "best_fitness": [],
    "mean_fitness": [],
}

if use_best_individual:
    save_best_individual["best_fitness"] = best_individual["best_fitness"]
    save_best_individual["mean_fitness"] = best_individual["mean_fitness"]

for i in range(100):
    evolution.step_generation()

    save_best_individual["best_fitness"].append(evolution.get_best_individual_fitness())
    save_best_individual["mean_fitness"].append(evolution.get_mean_fitness())

    print(
        len(save_best_individual["best_fitness"]),
        save_best_individual["best_fitness"][-1],
        evolution.get_mean_fitness(),
    )

    best_individual = evolution.get_best_individual()

    save_best_individual["lenia_params"] = best_individual
    with open("best_individual8", "wb") as f:
        pickle.dump(save_best_individual, f)
