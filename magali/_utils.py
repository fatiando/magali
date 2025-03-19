# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#

from ._constants import MICROMETER_TO_METER


def convet_micrometer_to_meter(coordinates):
    """
    Convert coordinates from micrometers to meters.

    Parameters:
    coordinates (tuple of float): A tuple containing coordinate values in micrometers.

    Returns:
    tuple of float: A tuple containing coordinate values converted to meters.
    """

    return tuple(c * MICROMETER_TO_METER for c in coordinates)
