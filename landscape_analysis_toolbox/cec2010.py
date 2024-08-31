from abc import ABC, abstractmethod

import numpy as np
from scipy.io import loadmat

MIN_PROBLEM_ID = 1
MAX_PROBLEM_ID = 18


def read_o_for_problem_id(problem_idx: int) -> np.ndarray:
    path = f"../cec2010/Function{problem_idx}.mat"
    return loadmat(path)["o"][0]


class CEC2010Problem(ABC):
    def __init__(self, problem_id: int, bounds: np.ndarray) -> None:
        dim = bounds.shape[0]
        assert MIN_PROBLEM_ID <= problem_id <= MAX_PROBLEM_ID, "Problem ID must be between 1 and 18."
        self.problem_id = problem_id
        self.o = read_o_for_problem_id(problem_id)[:dim]
        self.bounds = bounds
        self.dim = dim
        self.eps = 1e-4

    @abstractmethod
    def fitness(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the fitness value for the given input x.

        :param x: A 2D numpy array - solutions.
        :return: A 1D numpy array with fitness values.
        """
        pass

    @abstractmethod
    def constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the constraints for the given input x.

        :param x: A 2D numpy array - a row is a solution.
        :return: A 2D numpy array where each row contains the constraint values [g1, g2, ..., gN].
        """
        pass

    def violation(self, x: np.ndarray) -> np.ndarray:
        g = self.constraints(x)
        num_constraints = g.shape[1]
        return np.sum(np.maximum(0, g), axis=1) / num_constraints

    def _adjust_equality_constraint(self, h: np.ndarray) -> np.ndarray:
        # Convert equality constraint to inequality constraint:
        # h(x) = 0 -> |h(x)| - eps <= 0
        h_adjusted = np.abs(h) - self.eps
        return h_adjusted


class CEC2010Problem1(CEC2010Problem):
    def __init__(self, dim: int) -> None:
        super().__init__(problem_id=1, bounds=np.array([[0, 10]] * dim))

    def fitness(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - self.o
        f1 = np.sum(np.cos(x_shifted) ** 4, axis=1) - 2 * np.prod(np.cos(x_shifted) ** 2, axis=1)

        f2 = np.sum((np.arange(1, x.shape[1] + 1) * x_shifted**2), axis=1)

        fitness_values = -np.abs(f1 / np.sqrt(f2))
        return fitness_values

    def constraints(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - self.o

        g1 = 0.75 - np.prod(x_shifted, axis=1)
        g2 = np.sum(x_shifted, axis=1) - 7.5 * x.shape[1]

        g = np.column_stack((g1, g2))
        return g


class CEC2010Problem02(CEC2010Problem):
    def __init__(self, dim: int) -> None:
        super().__init__(problem_id=2, bounds=np.array([[-5.12, 5.12]] * dim))

    def fitness(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - self.o
        f = np.max(x_shifted, axis=1)
        return f

    def constraints(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - self.o
        y = x_shifted - 0.5

        g1 = 10 - np.sum(x_shifted**2 - 10 * np.cos(2 * np.pi * x_shifted) + 10, axis=1) / self.dim

        g2 = (np.sum(x_shifted**2 - 10 * np.cos(2 * np.pi * x_shifted) + 10, axis=1)) / self.dim - 15

        h = np.sum(y**2 - 10 * np.cos(2 * np.pi * y) + 10, axis=1) / self.dim - 20
        g3 = self._adjust_equality_constraint(h)

        g = np.column_stack((g1, g2, g3))
        return g


class CEC2010Problem3(CEC2010Problem):
    def __init__(self, dim: int) -> None:
        super().__init__(problem_id=3, bounds=np.array([[-1000, 1000]] * dim))

    def fitness(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - self.o
        f = np.sum(
            100 * (x_shifted[:, :-1] ** 2 - x_shifted[:, 1:]) ** 2 + (x_shifted[:, :-1] - 1) ** 2,
            axis=1,
        )
        return f

    def constraints(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - self.o
        h = np.sum((x_shifted[:, :-1] - x_shifted[:, 1:]) ** 2, axis=1)
        h_adjusted = self._adjust_equality_constraint(h)
        return h_adjusted[:, np.newaxis]


class CEC2010Problem4(CEC2010Problem):
    def __init__(self, dim: int) -> None:
        super().__init__(problem_id=4, bounds=np.array([[-50, 50]] * dim))

    def fitness(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - self.o
        f = np.max(x_shifted, axis=1)
        return f

    def constraints(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - self.o

        h1 = np.sum(x_shifted * np.cos(np.sqrt(np.abs(x_shifted))), axis=1) / self.dim
        h1_adjusted = self._adjust_equality_constraint(h1)

        half_dim = self.dim // 2
        h2 = np.sum(
            (x_shifted[:, : half_dim - 1] - x_shifted[:, 1:half_dim]) ** 2,
            axis=1,
        )
        h2_adjusted = self._adjust_equality_constraint(h2)

        h3 = np.sum((x_shifted[:, half_dim:-1] ** 2 - x_shifted[:, half_dim + 1 :]) ** 2, axis=1)  # noqa: E203
        h3_adjusted = self._adjust_equality_constraint(h3)

        h4 = np.sum(x_shifted, axis=1)
        h4_adjusted = self._adjust_equality_constraint(h4)

        return np.column_stack((h1_adjusted, h2_adjusted, h3_adjusted, h4_adjusted))


class CEC2010Problem13(CEC2010Problem):
    def __init__(self, dim: int) -> None:
        super().__init__(problem_id=13, bounds=np.array([[-500, 500]] * dim))

    def fitness(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - self.o
        f = -np.sum(-x_shifted * np.sin(np.sqrt(np.abs(x_shifted))), axis=1) / self.dim
        return f

    def constraints(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - self.o

        g1 = -50 + (1 / (100 * self.dim)) * np.sum(x_shifted**2, axis=1)

        g2 = (50 / self.dim) * np.sum(np.sin(1 / 50 * np.pi * x_shifted), axis=1)

        g3 = 75 - 50 * (
            np.sum(x_shifted**2, axis=1) / 4000 - np.prod(np.cos(x_shifted / np.sqrt(np.arange(1, self.dim + 1)))) + 1
        )

        g = np.column_stack((g1, g2, g3))
        return g
