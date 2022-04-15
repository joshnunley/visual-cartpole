import numpy as np


def MSE(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def cosine_fitness(x, y):
    x = x / (np.linalg.norm(x + 0.00001, axis=3, keepdims=True))
    y = y / (np.linalg.norm(y + 0.00001, axis=3, keepdims=True))

    return np.mean(np.sum(np.multiply(x, y), axis=3))


def correlation_fitness(x_input, y_input):
    x = np.copy(x_input)
    y = np.copy(y_input)
    x += 0.0001 * np.random.uniform(size=np.shape(x))
    y += 0.0001 * np.random.uniform(size=np.shape(y))

    centered_x = x - np.mean(x, axis=-1, keepdims=True)
    centered_y = y - np.mean(y, axis=-1, keepdims=True)
    return np.mean(
        np.sum(centered_x * centered_y, axis=-1)
        / (
            np.sqrt(
                np.sum(np.multiply(centered_x, centered_x), axis=-1)
                * np.sum(np.multiply(centered_y, centered_y), axis=-1)
            )
        )
    )
