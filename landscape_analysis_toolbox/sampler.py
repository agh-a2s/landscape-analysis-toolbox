import numpy as np


def get_uniform_samples(n_samples: int, bounds: np.ndarray) -> np.ndarray:
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
    assert lower_bounds.shape == upper_bounds.shape, "lower_bounds and upper_bounds must have the same shape"
    return np.random.uniform(lower_bounds, upper_bounds, (n_samples, lower_bounds.shape[0]))
