# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the _utils functions
"""

import numpy as np
import verde as vd
import xarray as xr

from .._utils import (
    _convert_micrometer_to_meter,
    _estimate_grid_spacing,
    gradient,
    total_gradient_amplitude,
    total_gradient_amplitude_grid,
)
from ._models import simple_model, souza_junior_model


def test_convert_micrometer_to_meter():
    coordinates_micrometer = vd.grid_coordinates(
        region=[0, 2000, 0, 2000],  # µm
        spacing=2,  # µm
    )
    coordinates_m = _convert_micrometer_to_meter(coordinates_micrometer)

    assert len(coordinates_m) == 2

    _convert_micrometer_to_meter(coordinates_micrometer)[0][0][1] == 2e-6


def test_estimate_grid_spacing(souza_junior_model):
    # Use model fixture from _models.py
    spacing = _estimate_grid_spacing(souza_junior_model)

    assert spacing == 2.0


def test_gradient(souza_junior_model):
    # Use model fixture from _models.py
    dx, dy, dz = gradient(souza_junior_model)

    np.testing.assert_allclose(dx.min().values, -92762.44269656, rtol=1e2)
    np.testing.assert_allclose(dy.min().values, -88701.89017599083, rtol=1e2)
    np.testing.assert_allclose(dz.min().values, -122948.1064382476, rtol=1e2)

    np.testing.assert_allclose(dx.max().values, 120507.65315237164, rtol=1e2)
    np.testing.assert_allclose(dy.max().values, 148905.76425923014, rtol=1e2)
    np.testing.assert_allclose(dz.max().values, 355923.5295921887, rtol=1e2)

    np.testing.assert_allclose(dx.std().values, 886.4487990775598, rtol=1e2)
    np.testing.assert_allclose(dy.std().values, 1164.8957138051455, rtol=1e2)
    np.testing.assert_allclose(dz.std().values, 1640.5332919277892, rtol=1e2)


def test_total_gradient_amplitude(souza_junior_model):
    # Use model fixture from _models.py
    data = souza_junior_model

    dx, dy, dz = gradient(data)

    tga = total_gradient_amplitude(dx, dy, dz)

    np.testing.assert_allclose(tga.min().values, 0.0013865457298526999, rtol=1e5)
    np.testing.assert_allclose(tga.max().values, 356478.0516202345, rtol=1e5)
    np.testing.assert_allclose(tga.std().values, 2192.1091330575123, rtol=1e5)


def test_total_gradient_amplitude_grid(souza_junior_model):
    # Use model fixture from _models.py
    data = souza_junior_model

    data_tga = total_gradient_amplitude_grid(data)

    # Test units
    assert data_tga.x.units == "µm"
    assert data_tga.y.units == "µm"
    assert data_tga.units == "nT/µm"

    # Test array sizes
    assert data_tga.x.size == 1001
    assert data_tga.y.size == 1001
    assert data_tga.size == 1002001

    # Test data name
    assert data_tga.long_name == "Total Gradient Amplitude"

    # Test if data is a DataArray
    assert isinstance(data_tga.x, xr.DataArray)
    assert isinstance(data_tga.y, xr.DataArray)
    assert isinstance(data_tga, xr.DataArray)
