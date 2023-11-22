#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt

from mark_distribution import MarkDistribution


def connection_kernel(
    first_marks: npt.NDArray[np.float32],
    second_marks: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Connection kernel function that scales the profile function argument.

    Args:
        first_marks (npt.NDArray[np.float32]): array with first marks
        second_marks (npt.NDArray[np.float32]): array with second marks

    Returns:
        npt.NDArray[np.float32]: N x 1 array with values of the connection
        kernel function
    """
    a = 3.

    # assert first_marks.ndim == 1 and second_marks.ndim == 1
    # assert len(first_marks) == len(second_marks)

    marks = np.c_[first_marks, second_marks]
    small_marks = np.min(marks, axis=1)
    big_marks = np.max(marks, axis=1)
    kernel_values = small_marks * big_marks**a
    return kernel_values


def calc_connection_kernel_conditional_expectation(
    first_mark: float,
) -> npt.NDArray[np.float32]:
    """Calculate the expectation of the connection kernel while keeping the
    first argument fixed.

    Args:
        first_mark (float): first argument of the kernel that is kept fixed

    Returns:
        npt.NDArray[np.float32]: Expectations for each of the provided weights
        taken as fixed first value
    """
    def callable(marks: npt.NDArray[np.float32]):
        return connection_kernel(np.full_like(marks, first_mark), marks)

    mark_distribution = MarkDistribution(a=0.)
    expectation = mark_distribution.vectorized_expectation(callable)
    return expectation


def is_connected(
    coordinates_1: npt.NDArray[np.float32],
    coordinates_2: npt.NDArray[np.float32],
    marks_1: npt.NDArray[np.float32],
    marks_2: npt.NDArray[np.float32],
    scaling
) -> bool:

    radius = np.sqrt(np.sum((coordinates_1 - coordinates_2)**2, axis=1))
    dimension = coordinates_1.shape[1]
    ball_volume = radius**dimension * np.pi**(dimension / 2.) / np.math.gamma(dimension / 2. + 1)

    profile_function_argument = \
        ball_volume / (scaling * connection_kernel(marks_1, marks_2))

    connection_probability = profile_function(profile_function_argument)

    return np.random.rand() < connection_probability


def profile_function(argument: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Profile function converting its argument to probability

    Args:
        argument (float): argument of the profile function

    Returns:
        float: connection probability
    """
    return np.where(argument < 1., 1., 0.)


def calc_typical_point_connections(
    coordinates: npt.NDArray[np.float32],
    marks: npt.NDArray[np.float32],
    typical_mark: float,
    scaling: float,
) -> npt.NDArray[np.bool_]:
    typical_coordinates = np.zeros_like(coordinates)
    typical_marks = np.full_like(marks, typical_mark)
    connections = is_connected(typical_coordinates, coordinates, typical_marks, marks, scaling)
    return connections
