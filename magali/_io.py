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
    """
    contents = scipy.io.loadmat(path)
    # For some reason, the spacing is returned as an array with a single
    # value. That messes up operations below so get the only element out.
    spacing = contents["step"].ravel()[0] * METER_TO_MICROMETER
    bz = contents["Bz"] * TESLA_TO_NANOTESLA
    sensor_sample_distance = contents["h"] * METER_TO_MICROMETER
    x = np.arange(bz.shape[1]) * spacing
    y = np.arange(bz.shape[0]) * spacing
    z = np.full(bz.shape, sensor_sample_distance)
    data = vd.make_xarray_grid(
        (x, y, z),
        bz,
        data_names=["bz"],
        dims=("y", "x"),
        extra_coords_names="z",
    )
    data.x.attrs = {"units": "µm"}
    data.y.attrs = {"units": "µm"}
    data.z.attrs = {"long_name": "sensor sample distance", "units": "µm"}
    data.bz.attrs = {"long_name": "vertical magnetic field", "units": "nT"}
    data.attrs = {"file_name": str(path)}
    return data
