# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#

import harmonica as hm
import numpy as np

from ._constants import MICROMETER_TO_METER


def _convert_micrometer_to_meter(coordinates):
    """
    Convert coordinates from micrometers to meters.

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


def _estimate_grid_spacing(data):
    """
    Estimate grid spacing as the mean difference in x and y coordinates.

    Parameters
    ----------
    data : xr.DataArray
        Input data array with coordinates "x" and "y".

    Returns
    -------
    spacing : float
        Estimated grid spacing.
    """
    return np.mean([np.abs(data.x[1] - data.x[0]), np.abs(data.y[1] - data.y[0])])


def gradient(data):
    """
    Compute first-order spatial derivatives in the x, y, and z directions.

    Parameters
    ----------
    data : xr.DataArray
        Input data array with coordinates "x" and "y".

    Returns
    -------
    dx : xr.DataArray
        First derivative along the x-direction.
    dy : xr.DataArray
        First derivative along the y-direction.
    dz : xr.DataArray
        First derivative along the z-direction.

    Notes
    -----
    The vertical derivative is estimated using the difference between an
    upward-continued and a downward-continued version of the data. This avoids
    downward continuation, which can amplify noise.
    """
    # Compute horizontal derivatives
    dx = hm.derivative_easting(data)
    dy = hm.derivative_northing(data)

    # Estimate grid spacing
    spacing = _estimate_grid_spacing(data)

    # Compute vertical derivative
    data_up = hm.upward_continuation(data, spacing).assign_coords(x=data.x, y=data.y)
    data_down = hm.upward_continuation(data, -spacing).assign_coords(x=data.x, y=data.y)
    dz = (data_up - data_down) / (2 * spacing)

    return dx, dy, dz


def total_gradient_amplitude(dx, dy, dz):
    """
    Compute the total gradient amplitude from spatial derivatives.

    Parameters
    ----------
    dx : xr.DataArray
        First derivative along the x-direction.
    dy : xr.DataArray
        First derivative along the y-direction.
    dz : xr.DataArray
        First derivative along the z-direction.

    Returns
    -------
    tga : xr.DataArray
        Total gradient amplitude.
    """
    return np.sqrt(dx**2 + dy**2 + dz**2)


def total_gradient_amplitude_grid(data):
    """
    Compute the total gradient amplitude (TGA) of a given data array.

    The function calculates the horizontal and vertical derivatives of the input data and then
    computes the total gradient amplitude.

    Parameters
    ----------
    data : xr.DataArray
        Input data array with coordinates `x` and `y`.

    Returns
    -------
    tga : xr.DataArray
        Dataset containing the total gradient amplitude (TGA).
    """
    dx, dy, dz = gradient(data)
    tga = total_gradient_amplitude(dx, dy, dz)

    # Assign attributes
    tga.attrs = {"long_name": "Total Gradient Amplitude", "units": "nT/µm"}

    return tga
