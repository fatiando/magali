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
    # For some reason, the spacing is returned as an array with a single
    # value. That messes up operations below so get the only element out.
    spacing = contents["step"].ravel()[0] * METER_TO_MICROMETER
    bz = contents["Bz"] * TESLA_TO_NANOTESLA
    data_names = ["bz"]
    shape = bz.shape
    sensor_sample_distance = contents["h"] * METER_TO_MICROMETER
    x = np.arange(shape[1]) * spacing
    y = np.arange(shape[0]) * spacing
    z = np.full(shape, sensor_sample_distance)
    data = vd.make_xarray_grid(
        (x, y, z),
        bz,
        data_names=data_names,
        dims=("y", "x"),
        extra_coords_names="z",
    )
    data.x.attrs = {"units": "µm"}
    data.y.attrs = {"units": "µm"}
    data.z.attrs = {"long_name": "sensor sample distance", "units": "µm"}
    data.bz.attrs = {"long_name": "vertical magnetic field", "units": "nT"}
    data.attrs = {"file_name": str(path)}
    return data
