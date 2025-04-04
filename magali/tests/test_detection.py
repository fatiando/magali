# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the _detection functions
"""

import harmonica as hm
import numpy as np
import skimage.exposure
import xarray as xr

from .._detection import detect_anomalies
from .._synthetic import dipole_bz_grid, random_directions
from .._utils import total_gradient_amplitude_grid


def test_detect_anomalies():
    # Synthetic
    sensor_sample_distance = 5.0  # µm
    region = [0, 2000, 0, 2000]  # µm
    spacing = 2  # µm

    dipole_coordinates = (
        np.asarray([1250, 1300, 500]),  # µm
        np.asarray([500, 1750, 1000]),  # µm
        np.asarray([-15, -15, -30]),  # µm
    )
    dipole_moments = hm.magnetic_angles_to_vec(
        inclination=np.asarray([10, -10, -5]),
        declination=np.asarray([10, 170, 190]),
        intensity=np.asarray([5e-11, 5e-11, 5e-11]),
    )

    data = dipole_bz_grid(
        region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moments
    )

    data_tga = total_gradient_amplitude_grid(data)
    stretched = skimage.exposure.rescale_intensity(
        data_tga,
        in_range=tuple(np.percentile(data_tga, (1, 99))),
    )
    data_tga_stretched = xr.DataArray(stretched, coords=data_tga.coords)

    # Detection
    windows = detect_anomalies(
        data_tga_stretched,
        size_range=[25, 50],
        size_multiplier=2,
        num_scales=10,
        detection_threshold=0.01,
        overlap_ratio=0.5,
        border_exclusion=1,
    )

    assert len(windows) == 3
    assert len(windows[0]) == 4
