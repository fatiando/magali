# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#

from ._constants import MICROMETER_TO_METER


def _convert_micrometer_to_meter(coordinates):
    """
    Converts coordinates from micrometers to meters.

    Parameters
    ----------
    coordinates : tuple of float
        Coordinate values in micrometers (μm).

    Returns
    -------
    coordinates_m : tuple of float
        Coordinate values converted to meters (m).
    """

    return tuple(c * MICROMETER_TO_METER for c in coordinates)
