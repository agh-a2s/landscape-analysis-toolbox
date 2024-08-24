import numpy as np
from .utils import apply_bounds


def progressive_random_walk(
    bounds: np.ndarray, num_steps: int, step_size: float, starting_zone: np.ndarray
) -> np.ndarray:
    """
    @inproceedings{malan2014progressive,
        title={A progressive random walk algorithm for sampling continuous fitness landscapes},
        author={Malan, Katherine M and Engelbrecht, Andries P},
        booktitle={2014 IEEE Congress on Evolutionary Computation (CEC)},
        pages={2507--2514},
        year={2014},
        organization={IEEE}
    }
    """
    n = len(bounds)
    lower, upper = bounds[:, 0], bounds[:, 1]
    walk = np.zeros((num_steps + 1, n), dtype=float)

    # Initialize the starting position
    r = np.random.uniform(0, (upper - lower) / 2, n)
    walk[0] = np.where(starting_zone, upper - r, lower + r)
    r_d = np.random.randint(0, n)
    walk[0, r_d] = upper[r_d] if starting_zone[r_d] else lower[r_d]

    for s in range(1, num_steps + 1):
        step = np.random.uniform(0, step_size, n)
        step[starting_zone] *= -1
        new_position = walk[s - 1] + step
        out_of_bounds = (new_position < lower) | (new_position > upper)
        if out_of_bounds.any():
            new_position = apply_bounds(new_position, bounds, "reflect")
            starting_zone ^= out_of_bounds

        walk[s] = new_position

    return walk


def simple_random_walk(
    bounds: np.ndarray, num_steps: int, step_size: float
) -> np.ndarray:
    """
    @inproceedings{malan2014progressive,
        title={A progressive random walk algorithm for sampling continuous fitness landscapes},
        author={Malan, Katherine M and Engelbrecht, Andries P},
        booktitle={2014 IEEE Congress on Evolutionary Computation (CEC)},
        pages={2507--2514},
        year={2014},
        organization={IEEE}
    }
    """
    n = len(bounds)
    lower, upper = bounds[:, 0], bounds[:, 1]
    walk = np.zeros((num_steps + 1, n), dtype=float)

    walk[0] = np.random.uniform(lower, upper)

    for s in range(1, num_steps + 1):
        for i in range(n):
            while True:
                r = np.random.uniform(-step_size, step_size)
                new_position = walk[s - 1, i] + r
                if lower[i] <= new_position <= upper[i]:
                    walk[s, i] = new_position
                    break

    return walk
