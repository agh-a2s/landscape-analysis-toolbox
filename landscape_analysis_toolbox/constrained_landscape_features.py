from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr

from .random_walk import get_starting_zones, progressive_random_walk
from .sampler import get_uniform_samples


def ratio_feasible_boundary_crossings(
    bounds: np.ndarray,
    violation: Callable[[np.ndarray], float],
    n_walks: int = 30,
    num_steps: int = 1000,
    step_size: float | None = None,
):
    crossings = []
    starting_zones = get_starting_zones(n_walks)
    if step_size is None:
        step_size = (bounds[:, 1] - bounds[:, 0]) / 100
    for starting_zone in starting_zones:
        walk = progressive_random_walk(bounds, num_steps, step_size, starting_zone=starting_zone)
        binary_violations = np.array([1 if violation(x) > 0 else 0 for x in walk])
        crossings.append(np.sum(np.abs(np.diff(binary_violations))) / (num_steps - 1))
    return crossings


def feasibility_ratio(samples: np.ndarray, violation_values: np.ndarray):
    return np.mean([y > 0 for _, y in zip(samples, violation_values)])


def fitness_violation_correlation(fitness_values: np.ndarray, violation_values: np.ndarray):
    return spearmanr(fitness_values, violation_values)[0]


def proportion_in_ideal_zone(
    fitness_values: np.ndarray,
    violation_values: np.ndarray,
    quantile: float,
):
    f_threshold = np.quantile(fitness_values, quantile)
    v_threshold = np.quantile(violation_values, quantile)

    return np.mean((fitness_values <= f_threshold) & (violation_values <= v_threshold))


class ConstrainedLandscapeFeatures:
    # TODO: Test it on CEC 2010 benchmarks
    """
    @inproceedings{malan2015characterising,
        title={Characterising constrained continuous optimisation problems},
        author={Malan, Katherine M and Oberholzer, Johannes F and Engelbrecht, Andries P},
        booktitle={2015 IEEE Congress on Evolutionary Computation (CEC)},
        pages={1351--1358},
        year={2015},
        organization={IEEE}
    }
    """

    def __init__(
        self,
        bounds: np.ndarray,
        fitness: Callable[[np.ndarray], float],
        violation: Callable[[np.ndarray], float],
        sampler: Callable[[int, np.ndarray], np.ndarray] = get_uniform_samples,
    ):
        self.bounds = bounds
        self.violation = violation
        self.fitness = fitness
        self.sampler = sampler

    def compute_features(
        self,
        num_samples: int | None = None,
        num_runs: int = 30,
        num_steps: int = 1000,
        step_size: float | None = None,
        quantiles: list[float] = [0.01, 0.25, 0.5],
    ) -> pd.DataFrame:
        if num_samples is None:
            num_samples = self._get_default_num_samples()
        all_features = []
        # To calculate the ratio of feasible boundary crossings, we need to run the random walks.
        # It's independent from other features (that require uniform sampling), so we calculate it first.
        crossings = ratio_feasible_boundary_crossings(self.bounds, self.violation, num_runs, num_steps, step_size)
        for run_idx in range(num_runs):
            uniform_samples = self.sampler(num_samples, self.bounds)
            fitness_values = np.array([self.fitness(x) for x in uniform_samples])
            violation_values = np.array([self.violation(x) for x in uniform_samples])
            ideal_zones = {
                f"PiIZ_{q}": proportion_in_ideal_zone(fitness_values, violation_values, q) for q in quantiles
            }
            crossing = crossings[run_idx]
            features = {
                "FsR": feasibility_ratio(uniform_samples, violation_values),
                "FVC": fitness_violation_correlation(fitness_values, violation_values),
                "RFB_x": crossing,
            } | ideal_zones
            all_features.append(features)
        return pd.DataFrame(all_features)

    def plot_fitness_validation(self, num_samples: int | None = None) -> None:
        if num_samples is None:
            num_samples = self._get_default_num_samples()
        uniform_samples = self.sampler(num_samples, self.bounds)
        fitness_values = np.array([self.fitness(x) for x in uniform_samples])
        violation_values = np.array([self.violation(x) for x in uniform_samples])
        df = pd.DataFrame({"fitness": fitness_values, "violation": violation_values})
        fig = px.scatter(df, x="fitness", y="violation")
        fig.update_layout(
            title="Fitness vs. Violation",
            xaxis_title="Fitness",
            yaxis_title="Violation",
        )
        fig.show()

    def _get_default_num_samples(self) -> int:
        return 1000 * self.bounds.shape[0]
