# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Models that are used in tests
"""

import harmonica as hm
import numpy as np
import pytest

from .._synthetic import dipole_bz_grid, random_directions


@pytest.fixture
def simple_model():
    """
    A simple model containing 3 anomalies
    """
    # Synthetic
    sensor_sample_distance = 5.0  # µm
    region = [0, 2000, 0, 2000]  # µm
    spacing = 2  # µm

    dipole_coordinates = (
        np.asarray([1250, 1300, 500]),  # µm
        np.asarray([500, 1750, 1000]),  # µm
        np.asarray([-15, -15, -30]),  # µm
    )
    dipole_moments = hm.magnetic_angles_to_vec(
        inclination=np.asarray([10, -10, -5]),
        declination=np.asarray([10, 170, 190]),
        intensity=np.asarray([5e-11, 5e-11, 5e-11]),
    )

    return dipole_bz_grid(
        region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moments
    )


@pytest.fixture
def souza_junior_model():
    """
    A model simmilar to the one used by Souza-Junior
    """
    sensor_sample_distance = 5.0  # µm
    region = [0, 2000, 0, 2000]  # µm
    spacing = 2  # µm

    true_inclination = 30
    true_declination = 40
    true_dispersion_angle = 5

    size = 100

    directions_inclination, directions_declination = random_directions(
        true_inclination,
        true_declination,
        true_dispersion_angle,
        size=size,
        random_state=5,
    )

    dipoles_amplitude = abs(np.random.normal(0, 100, size)) * 1.0e-14

    dipole_coordinates = (
        np.concatenate([np.random.randint(30, 1970, size), [1250, 1300, 500]]),  # µm
        np.concatenate([np.random.randint(30, 1970, size), [500, 1750, 1000]]),  # µm
        np.concatenate([np.random.randint(-20, -1, size), [-15, -15, -30]]),  # µm
    )
    dipole_moments = hm.magnetic_angles_to_vec(
        inclination=np.concatenate([directions_inclination, [10, -10, -5]]),
        declination=np.concatenate([directions_declination, [10, 170, 190]]),
        intensity=np.concatenate([dipoles_amplitude, [5e-11, 5e-11, 5e-11]]),
    )

    return dipole_bz_grid(
        region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moments
    )
