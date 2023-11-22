#!/usr/bin/env python3

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.stats import rv_continuous


left_right_limit = 1.
p = 0.5
rho = 2.
beta = 10.


class MarkDistribution(rv_continuous):
    """Distribution of the marks for the points.

    Args:
        rv_continuous (class): base class for continuous random variables
    """

    def vectorized_expectation(
        self,
        callable: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
    ) -> npt.NDArray[np.float32]:
        """Calculate the expectation of the connection kernel while keeping the
        first argument fixed.

        Args:
            first_mark (float): first argument of the kernel that is kept fixed

        Returns:
            npt.NDArray[np.float32]: Expectations for each of the provided weights
            taken as fixed first value
        """
        upper_bound = 1000.
        num_of_points = int(1e6)

        marks = np.linspace(0., upper_bound, num_of_points, endpoint=True)
        pdf_values = self.pdf(marks)
        values = callable(marks)

        delta_x = marks[1] - marks[0]
        expectation = np.sum(values * pdf_values) * delta_x
        return expectation

    def _rvs(self, size=None, random_state=None) -> npt.NDArray[np.float32]:
        if size is None:
            size = 1
        r = np.random.uniform(size=size)
        random_variables = np.where(
            r < p * left_right_limit**rho,
            (r/p)**(1./rho),
            (left_right_limit**(-beta) - (r - p * left_right_limit**rho) / (1. - p))**(-1./beta),
        )
        return random_variables

    def _pdf(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Probability density function for the mark distribution.

        Args:
            x (npt.NDArray[np.float32]): values of the random variable

        Returns:
            npt.NDArray[np.float32]: the PDF evaluated at the given x values
        """
        assert beta > 1.

        normalization = p * left_right_limit**rho + (1. - p) * left_right_limit**(-beta)
        pdf = np.where(
            x < left_right_limit,
            self._pdf_left_tail(x, p, rho),
            self._pdf_right_tail(x, p, beta)
        ) / normalization

        return pdf

    def _pdf_left_tail(self, x, p: float, rho: float) -> npt.NDArray[np.float32]:
        # Define the PDF function for the left tail
        # You can replace this with your specific function
        return p * rho * x**(rho - 1.)

    def _pdf_right_tail(self, x, p: float, beta: float) -> npt.NDArray[np.float32]:
        # Define the PDF function for the right tail
        # You can replace this with your specific function
        return (1. - p) * beta * x**(- beta - 1.)
