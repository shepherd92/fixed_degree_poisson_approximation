#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mark_distribution import MarkDistribution


def plot_typical_mark_vs_intensity(table: pd.DataFrame) -> plt.figure:
    figure, axes = plt.subplots()
    axes.scatter(table['intensity'], table['typical_mark'], c='black', s=0.3)
    axes.set_xscale('log')
    axes.set_yscale('log')
    # intensity_points = np.linspace(table['intensity'].min(), table['intensity'].max(), 1000)
    return figure


def plot_points(coordinates, marks, typical_mark: float, domain_dimensions) -> plt.figure:
    figure, axes = plt.subplots()

    dimension = coordinates.shape[1]
    if dimension == 1:
        axes.scatter(coordinates[:, 0], marks, c='black', s=0.3)
        axes.scatter([0], typical_mark, c='red', s=10)
        axes.set_xlabel('coordinate')
        axes.set_ylabel('mark')
        axes.set_xlim([
            -0.5 * domain_dimensions[0] * 1.05,
            +0.5 * domain_dimensions[0] * 1.05,
        ])
        axes.axvline(x=-0.5 * domain_dimensions[0], color='black', linewidth=0.3)
        axes.axvline(x=+0.5 * domain_dimensions[0], color='black', linewidth=0.3)
        axes.axhline(y=0, color='black', linewidth=0.3)

        mark_distribution = MarkDistribution(a=0.0)
        mark_max_percentile = mark_distribution.ppf(0.999)
        axes.set_ylim([-0.05, mark_max_percentile])
    elif dimension == 2:
        axes.scatter(coordinates[:, 0], coordinates[:, 1], s=np.log(marks))
        axes.set_xlim([
            -0.5 * domain_dimensions[0] * 1.05,
            +0.5 * domain_dimensions[0] * 1.05,
        ])
        axes.set_ylim([
            -0.5 * domain_dimensions[1] * 1.05,
            +0.5 * domain_dimensions[1] * 1.05,
        ])
        axes.axvline(x=-0.5 * domain_dimensions[0], color='black', linewidth=0.3)
        axes.axvline(x=+0.5 * domain_dimensions[0], color='black', linewidth=0.3)
        axes.axhline(y=-0.5 * domain_dimensions[1], color='black', linewidth=0.3)
        axes.axhline(y=+0.5 * domain_dimensions[1], color='black', linewidth=0.3)

    return figure
