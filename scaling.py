#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize


from connection import calc_connection_kernel_conditional_expectation
from mark_distribution import MarkDistribution


def determine_scaling(s: float, k: int, p: float, rho: float, a: float) -> float:
    mark_distribution = MarkDistribution(a=0.)
    moment_a = mark_distribution.expect(lambda x: x**a)

    scaling = 1. / (s * moment_a) * (
        s * p * rho * np.math.gamma(k + rho) / np.math.factorial(k)
    )**(1. / rho)

    return scaling


def optimize_scaling(intensity: float, k: int) -> float:

    def cost_function(scaling: npt.NDArray[np.float32]) -> float:
        expectation = determine_expectation_for_scaling(scaling[0], intensity, k)
        goal = np.math.factorial(k) / intensity
        cost = (goal - expectation)**2
        print(f'scaling: {scaling[0]:.6f}; cost: {cost:.6f}')
        return cost

    initial_value = intensity
    eps = 1. * intensity
    result = minimize(cost_function, x0=initial_value, method='BFGS', options={'eps': eps})

    return result.x


def determine_expectation_for_scaling(scaling: float, intensity: float, k: int) -> float:

    def expectation_argument(weight: float) -> float:
        s = intensity
        v = scaling
        h = calc_connection_kernel_conditional_expectation(weight)
        return (s * v * h)**k * np.exp(- s * v * h)

    mark_distribution = MarkDistribution(a=0.)
    expectation = mark_distribution.expect(expectation_argument)
    return expectation
