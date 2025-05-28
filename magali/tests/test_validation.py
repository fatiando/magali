# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the input validation functions
"""

import pytest

from .._validation import check_coordinates, check_fit_input


@pytest.mark.parametrize(
    ("coordinates", "data"),
    [
        (([1, 2], [3, 4], [5, 6]), [7, 8]),
        (([[1], [2]], [[3], [4]], [[5], [6]]), [[7], [8]]),
    ],
)
def test_check_fit_input_passes(coordinates, data):
    "Make sure valid inputs don't cause errors"
    coordinates_val, data_val = check_fit_input(coordinates, data)
    assert all(c.shape == data_val.shape for c in coordinates_val)
    assert all(len(c.shape) == 1 for c in coordinates_val)
    assert len(data_val.shape) == 1


@pytest.mark.parametrize(
    ("coordinates", "data", "message"),
    [
        (([1, 2], [3, 4]), [7, 8], "Three coordinates are required"),
        (([1, 2], [3, 4], [5, 6], [7, 8]), [7, 8], "Three coordinates are required"),
        (([[1], [2]], [[3], [4]], [5, 6]), [[7], [8]], "Invalid coordinate arrays"),
        (([1, 2], [3, 4], [5, 6, 7]), [[7], [8]], "Invalid coordinate arrays"),
        (([1, 2], [3, 4], [5, 6]), [[7], [8]], "Data array must have the same shape"),
        (([1, 2], [3, 4], [5, 6]), [7, 8, 9], "Data array must have the same shape"),
    ],
)
def test_check_fit_input_fails(coordinates, data, message):
    "Make sure exceptions are raised for invalid inputs"
    with pytest.raises(ValueError, match=message):
        check_fit_input(coordinates, data)


@pytest.mark.parametrize(
    "coordinates",
    [
        ([1, 2], [3, 4], [5, 6]),
        ([[1], [2]], [[3], [4]], [[5], [6]]),
    ],
)
def test_check_coordinates_passes(coordinates):
    "Make sure valid inputs don't cause errors"
    coordinates_val = check_coordinates(coordinates)
    assert all(c.shape == coordinates_val[0].shape for c in coordinates_val)


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        (([1, 2], [3, 4]), "Three coordinates are required"),
        (([1, 2], [3, 4], [5, 6], [7, 8]), "Three coordinates are required"),
        (([[1], [2]], [[3], [4]], [5, 6]), "Invalid coordinate arrays"),
        (([1, 2], [3, 4], [5, 6, 7]), "Invalid coordinate arrays"),
    ],
)
def test_check_coordinates_fails(coordinates, message):
    "Make sure exceptions are raised for invalid inputs"
    with pytest.raises(ValueError, match=message):
        check_coordinates(coordinates)
