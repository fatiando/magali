# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utilities for converting between units we use a lot.
"""

TESLA_TO_NANOTESLA = 1e9
METER_TO_MICROMETER = 1e6


def coordinates_micrometer_to_meter(coordinates):
    """
    Convert coordinates from micrometers to meters.

    Parameters
    ----------
    coordinates : tuple = (x, y, z)
        Coordinate values in micrometers (μm). Coordinates must be arrays or
        floats/ints.

    Returns
    -------
    coordinates : tuple of float
        Coordinate values converted to meters (m).
    """
    return tuple(c / METER_TO_MICROMETER for c in coordinates)


def meter_to_micrometer(value):
    """
    Convert a value from meters to micrometers.

    Parameters
    ----------
    value : int, float, or array
        The value in meters to convert.

    Returns
    -------
    value : float or array
        The value converted to micrometers.
    """
    return value * METER_TO_MICROMETER


def nanotesla_to_tesla(value):
    """
    Convert a value from nanoTesla to Tesla.

    Parameters
    ----------
    value : int, float, or array
        The value in nanoTesla to convert.

    Returns
    -------
    value : float or array
        The value converted to Tesla.
    """
    return value / TESLA_TO_NANOTESLA


def tesla_to_nanotesla(value):
    """
    Convert a value from Tesla to nanoTesla.

    Parameters
    ----------
    value : int, float, or array
        The value in Tesla to convert.

    Returns
    -------
    value : float or array
        The value converted to nanoTesla.
    """
    return value * TESLA_TO_NANOTESLA
