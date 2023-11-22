#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from connection import calc_typical_point_connections
from mark_distribution import MarkDistribution
from scaling import determine_scaling


def generate_point_cloud(intensity: float, seed: int, output_dir: Path):
    domain_dimensions = np.array([1.])
    k = 0
    p = 0.5
    rho = 2.
    a = 3.
    num_of_required_successes = 1

    np.random.seed(seed=seed)
    scaling = determine_scaling(intensity, k, p, rho, a)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'coordinates').mkdir(parents=True, exist_ok=True)

    success = 0
    pbar = tqdm(leave=False, desc='tries')

    results_current_intensity: list[list[float]] = []
    while success < num_of_required_successes:
        coordinates, marks = generate_poisson_points(
            domain_dimensions=domain_dimensions,
            intensity=intensity
        )

        typical_mark = generate_poisson_point_marks(1)[0]
        while typical_mark > 100 * intensity**(-1/rho):
            typical_mark = generate_poisson_point_marks(1)[0]

        connections = calc_typical_point_connections(coordinates, marks, typical_mark, scaling)
        if connections.sum() == k:
            success += 1
            pbar.update()
            # plot = plot_points(coordinates, marks, typical_mark, domain_dimensions)
            # plot.savefig(output_dir / 'images' / f'plot_k_{k}_intensity_{intensity:4f}_{success}.png')
            pd.DataFrame(
                np.c_[coordinates, marks],
                columns=['coordinate', 'mark']
            ).to_csv(output_dir / 'coordinates' / f'points_k_{k}_intensity_{intensity:4f}_{success}.csv', index=False)
            results_current_intensity.append([intensity, typical_mark])

    pbar.close()
    return results_current_intensity


def generate_poisson_points(
    domain_dimensions: npt.NDArray[np.float32],
    intensity: float,
):
    coordinates = generate_poisson_point_coordinates(domain_dimensions, intensity)
    marks = generate_poisson_point_marks(len(coordinates))

    return coordinates, marks


def generate_poisson_point_coordinates(
    domain_dimensions: npt.NDArray[np.float32],
    intensity: float,
) -> npt.NDArray[np.float32]:
    """_summary_

    Returns:
        npt.NDArray[np.float32]: _description_
    """
    dimension = len(domain_dimensions)
    volume = np.prod(domain_dimensions)
    num_of_points = np.random.poisson(intensity * volume)

    coordinates = np.random.rand(num_of_points, dimension) - 0.5
    for d, size in enumerate(domain_dimensions):
        coordinates[:, d] = coordinates[:, d] * size

    # convert coordinates array to 2D
    # coordinates = coordinates[:, None] \
    #     if len(np.shape(coordinates)) == 1 \
    #     else coordinates

    return coordinates


def generate_poisson_point_marks(num_of_marks: int) -> npt.NDArray[np.float32]:

    mark_distribution = MarkDistribution(a=0.)
    marks = mark_distribution.rvs(size=num_of_marks)
    return marks
