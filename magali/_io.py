# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions to read data from instrument files
"""

import numpy as np
import scipy.io
import verde as vd

from ._constants import METER_TO_MICROMETER, TESLA_TO_NANOTESLA


def read_qdm_harvard(path):
    """
    Load QDM microscopy data in the Harvard group format.

    This is the file type used by Roger Fu's group to distribute QDM data. It's
    a Matlab binary file that has the data and some information about grid
    spacing.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input Matlab binary file.

    Returns
    -------
    data : xarray.Dataset
        The magnetic field data as a regular grid with coordinates. The
        coordinates are in µm and magnetic field in nT.
    """
    contents = scipy.io.loadmat(path)
    coordinates, data_names, bz = _extract_data_qdm_harvard(contents)
    data = _create_qdm_harvard_grid(coordinates, data_names, bz, path)
    return data


def _extract_data_qdm_harvard(contents):
    """
    Define variables for generating a grid from QDM microscopy data.

    Parameters
    ----------
    contents: dict
        A dictionary containing essential parameters including spacing, Bz
        component, and sensor sample distances.

    Returns
    -------
    coordinates: tuple of arrays
        Tuple of 1D arrays with coordinates of each point in the grid:
        x, y, and z (vertical).
    data_names : str or list
        The name(s) of the data variables in the output grid. Ignored if data
        is None.
    bz : array
        The Bz component in nT.
    """
    # For some reason, the spacing is returned as an array with a single
    # value. That messes up operations below so get the only element out.
    spacing = contents["step"].ravel()[0] * METER_TO_MICROMETER
    bz = contents["Bz"] * TESLA_TO_NANOTESLA
    data_names = ["bz"]
    sensor_sample_distance = contents["h"] * METER_TO_MICROMETER
    shape = bz.shape
    x = np.arange(shape[1]) * spacing
    y = np.arange(shape[0]) * spacing
    z = np.full(shape, sensor_sample_distance)
    return (x, y, z), data_names, bz


def _create_qdm_harvard_grid(coordinates, data_names, bz, path):
    """
    Creates QDM microscopy data in the Harvard group format.

    This functions makes the xarray.Dataset and sets appropriate metadata.

    Parameters
    ----------
    coordinates: tuple of arrays
        Arrays with coordinates of each point in the grid. Each array must
        contain values for a dimension in the order: easting, northing,
        vertical, etc. All arrays must be 2d and need to have the same shape.
        These coordinates can be generated through verde.grid_coordinates.
    data_names : str or list
        The name(s) of the data variables in the output grid. Ignored if data
        is None.
    path : str or pathlib.Path
        Path to the input Matlab binary file.
    bz : array
        The Bz component in nT.

    Returns
    -------
    qdm_data : xarray.Dataset
        The magnetic field data as a regular grid with coordinates. The
        coordinates are in µm and magnetic field in nT.
    """
    qdm_data = vd.make_xarray_grid(
        coordinates,
        bz,
        data_names=data_names,
        dims=("y", "x"),
        extra_coords_names="z",
    )
    qdm_data.x.attrs = {"units": "µm"}
    qdm_data.y.attrs = {"units": "µm"}
    qdm_data.z.attrs = {"long_name": "sensor sample distance", "units": "µm"}
    qdm_data.bz.attrs = {"long_name": "vertical magnetic field", "units": "nT"}
    qdm_data.attrs = {"file_name": str(path)}
    return qdm_data
