# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the unit conversion functions.
"""

import numpy as np
import numpy.testing as npt
import pytest

from magali._units import (
    coordinates_micrometer_to_meter,
    meter_to_micrometer,
    nanotesla_to_tesla,
    tesla_to_nanotesla,
)


@pytest.mark.parametrize(
    "coordinates",
    [
        (np.array([1, 1]), np.array([1, 1]), np.array([1, 1])),
        (1, 1, 1),
    ],
)
def test_coordinates_micrometer_to_meter(coordinates):
    "Check that conversion yields expected outputs"
    converted = coordinates_micrometer_to_meter(coordinates)
    npt.assert_allclose(converted, 1e-6)


@pytest.mark.parametrize(
    "value",
    [
        np.array([1, 1, 1, 1]),
        1,
    ],
)
def test_meter_to_micrometer(value):
    "Check that conversion yields expected outputs"
    converted = meter_to_micrometer(value)
    npt.assert_allclose(converted, 1e6)


@pytest.mark.parametrize(
    "value",
    [
        np.array([1, 1, 1, 1]),
        1,
    ],
)
def test_nanotesla_to_tesla(value):
    "Check that conversion yields expected outputs"
    converted = nanotesla_to_tesla(value)
    npt.assert_allclose(converted, 1e-9)


@pytest.mark.parametrize(
    "value",
    [
        np.array([1, 1, 1, 1]),
        1,
    ],
)
def test_tesla_to_nanotesla(value):
    "Check that conversion yields expected outputs"
    converted = tesla_to_nanotesla(value)
    npt.assert_allclose(converted, 1e9)
