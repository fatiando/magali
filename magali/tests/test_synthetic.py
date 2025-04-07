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
from .._synthetic import dipole_bz, random_directions
from ._models import simple_model, souza_junior_model


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

    np.testing.assert_allclose(bz.min(), -4.693868775099469e18, rtol=1e5)
    np.testing.assert_allclose(bz.max(), 1.0003971128213485e18, rtol=1e5)
    np.testing.assert_allclose(bz.mean(), -76460148158726.5, rtol=1e5)
    np.testing.assert_allclose(bz.std(), 3.918804767967805e16, rtol=1e5)
    np.testing.assert_allclose(bz.size, 1002001, rtol=1e5)


def test_dipole_bz_grid(souza_junior_model):
    # Use model fixture from _models.py
    data = souza_junior_model

    # Test units
    assert data.x.units == "µm"
    assert data.y.units == "µm"
    assert data.units == "nT"

    # Test array sizes
    assert data.x.size == 1001
    assert data.y.size == 1001
    assert data.size == 1002001

    # Test data name
    assert data.long_name == "vertical magnetic field"

    # Test if data is a DataArray
    assert isinstance(data.x, xr.DataArray)
    assert isinstance(data.y, xr.DataArray)
    assert isinstance(data, xr.DataArray)
