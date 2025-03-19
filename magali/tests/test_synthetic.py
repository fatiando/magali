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
import verde as vd
import xarray as xr

from .._stats import variance
from .._synthetic import dipole_bz, dipole_bz_grid, random_directions


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


def test_dipole_bz():
    sensor_sample_distance = 5.0  # µm

    coordinates = vd.grid_coordinates(
        region=[0, 2000, 0, 2000],  # µm
        spacing=2,  # µm
        extra_coords=sensor_sample_distance,
    )

    true_inclination = 30
    true_declination = 40
    true_dispersion_angle = 5

    size = 10

    directions_inclination, directions_declination = random_directions(
        true_inclination,
        true_declination,
        true_dispersion_angle,
        size=size,
        random_state=5,
    )

    amplitude = abs(np.random.normal(0, 100, size)) * 1.0e-14

    dipole_moments = hm.magnetic_angles_to_vec(
        directions_inclination, directions_declination, amplitude
    )

    dipole_coordinates = (
        np.random.randint(30, 1970, size),  # µm
        np.random.randint(30, 1970, size),  # µm
        np.random.randint(-20, -1, size),  # µm
    )

    bz = dipole_bz(coordinates, dipole_coordinates, dipole_moments)


def test_dipole_bz_grid():
    sensor_sample_distance = 5.0  # µm
    region = [0, 2000, 0, 2000]
    spacing = 2

    true_inclination = 30
    true_declination = 40
    true_dispersion_angle = 5

    size = 10

    directions_inclination, directions_declination = random_directions(
        true_inclination,
        true_declination,
        true_dispersion_angle,
        size=size,
        random_state=5,
    )

    amplitude = abs(np.random.normal(0, 100, size)) * 1.0e-14

    dipole_moments = hm.magnetic_angles_to_vec(
        directions_inclination, directions_declination, amplitude
    )

    dipole_coordinates = (
        np.random.randint(30, 1970, size),  # µm
        np.random.randint(30, 1970, size),  # µm
        np.random.randint(-20, -1, size),  # µm
    )

    data = dipole_bz_grid(
        region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moments
    )

    # Test units
    assert data.x.units == "µm"
    assert data.y.units == "µm"
    assert data.bz.units == "nT"

    # Test array sizes
    assert data.x.size == 1001
    assert data.y.size == 1001
    assert data.bz.size == 1002001

    # Test data name
    assert data.bz.long_name == "vertical magnetic field"

    # Test if data is a DataArray
    assert isinstance(data.x, xr.DataArray)
    assert isinstance(data.y, xr.DataArray)
    assert isinstance(data.bz, xr.DataArray)
