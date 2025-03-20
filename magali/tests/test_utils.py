# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the _utils functions
"""

import verde as vd
import numpy as np

from .._utils import _convert_micrometer_to_meter


def test_convert_micrometer_to_meter():
    coordinates_micrometer = vd.grid_coordinates(
        region=[0, 2000, 0, 2000],  # µm
        spacing=2,  # µm
    )
    coordinates_m = _convert_micrometer_to_meter(coordinates_micrometer)

    assert len(coordinates_m) == 2

    _convert_micrometer_to_meter(coordinates_micrometer)[0][0][1] == 2e-6