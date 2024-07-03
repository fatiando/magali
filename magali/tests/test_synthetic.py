# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the _synthetic functions
"""

from .._synthetic import random_directions


def test_random_directions():
    "Test values of inclination and declination for the random directions"
    directions_inclination, directions_declination = random_directions(
        30, 40, 5, 100, random_state=5
    )

    assert (float(directions_inclination[0]) == -18.382758083412796) and (
        float(directions_declination[0]) == -13.603233653244104
    )
