# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the _synthetic functions
"""

import harmonica as hm
import numpy as np

from .._stats import variance
from .._synthetic import (
    amplitude_lognormal_distribution,
    generate_dipoles_grid,
    random_coordinates,
    random_directions,
)


def test_random_directions():
    """
    Tests inclination and declination inputs. Also compares the variance of
    directions and compares to the dispersion angle input
    """
    true_inclination = 30
    true_declination = 40
    true_dispersion_angle = 5

    directions_inclination, directions_declination = random_directions(
        true_inclination,
        true_declination,
        true_dispersion_angle,
        size=1000000,
        random_state=5,
    )

    x, y, z = hm.magnetic_angles_to_vec(
        1, directions_inclination, directions_declination
    )

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    _, inclination_mean, declination_mean = hm.magnetic_vec_to_angles(
        x_mean, y_mean, z_mean
    )

    np.testing.assert_allclose(inclination_mean, true_inclination, rtol=1e-4)
    np.testing.assert_allclose(declination_mean, true_declination, rtol=1e-4)

    directions_variance = variance(
        directions_inclination,
        directions_declination,
        inclination_mean,
        declination_mean,
    )
    std_dev = np.sqrt(directions_variance)
    np.testing.assert_allclose(true_dispersion_angle, std_dev, rtol=1e-3)


def test_generate_dipoles_grid():
    region = [0, 2000, 0, 2000]
    spacing = 2
    sensor_sample_distance = 5

    number_of_sources = 1000
    sources_region = [30, 1970, 30, 1970, -20, -1]
    seed = 5

    coordinates = random_coordinates(sources_region, number_of_sources, seed)
    amplitude = amplitude_lognormal_distribution(0, 1.35, 1000)

    true_inclination = 30
    true_declination = 40
    true_dispersion_angle = 5

    directions_inclination, directions_declination = random_directions(
        true_inclination,
        true_declination,
        true_dispersion_angle,
        size=1000,
        random_state=5,
    )

    field = "b_u"

    grid = generate_dipoles_grid(
        region,
        spacing,
        sensor_sample_distance,
        coordinates,
        amplitude,
        directions_inclination,
        directions_declination,
        field,
    )

    np.testing.assert_allclose(grid.bz.size, 1002001, atol=1e-4)
