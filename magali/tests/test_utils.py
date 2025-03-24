# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the _utils functions
"""

import harmonica as hm
import numpy as np
import verde as vd

from .._synthetic import dipole_bz_grid, random_directions
from .._utils import (
    _convert_micrometer_to_meter,
    _estimate_grid_spacing,
    gradients,
    total_gradient_amplitude,
)


def test_convert_micrometer_to_meter():
    coordinates_micrometer = vd.grid_coordinates(
        region=[0, 2000, 0, 2000],  # µm
        spacing=2,  # µm
    )
    coordinates_m = _convert_micrometer_to_meter(coordinates_micrometer)

    assert len(coordinates_m) == 2

    _convert_micrometer_to_meter(coordinates_micrometer)[0][0][1] == 2e-6


def test_estimate_grid_spacing():
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

    data = dipole_bz_grid(
        region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moments
    )

    spacing = _estimate_grid_spacing(data)

    assert spacing == 2.0


def test_gradients():
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

    data = dipole_bz_grid(
        region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moments
    )

    dx, dy, dz = gradients(data)

    np.testing.assert_allclose(dx.min().values, -92762.44269656, rtol=1e2)
    np.testing.assert_allclose(dy.min().values, -88701.89017599083, rtol=1e2)
    np.testing.assert_allclose(dz.min().values, -122948.1064382476, rtol=1e2)

    np.testing.assert_allclose(dx.max().values, 120507.65315237164, rtol=1e2)
    np.testing.assert_allclose(dy.max().values, 148905.76425923014, rtol=1e2)
    np.testing.assert_allclose(dz.max().values, 355923.5295921887, rtol=1e2)

    np.testing.assert_allclose(dx.std().values, 886.4487990775598, rtol=1e2)
    np.testing.assert_allclose(dy.std().values, 1164.8957138051455, rtol=1e2)
    np.testing.assert_allclose(dz.std().values, 1640.5332919277892, rtol=1e2)


def test_total_gradient_amplitude():
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

    data = dipole_bz_grid(
        region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moments
    )

    dx, dy, dz = gradients(data)

    tga = total_gradient_amplitude(dx, dy, dz)


    np.testing.assert_allclose(tga.min().values, 0.0013865457298526999, rtol=1e5)
    np.testing.assert_allclose(tga.max().values, 356478.0516202345, rtol=1e5)
    np.testing.assert_allclose(tga.std().values, 2192.1091330575123, rtol=1e5)
