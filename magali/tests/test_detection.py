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
from .._utils import total_gradient_amplitude_grid
from ._models import simple_model



def test_detect_anomalies(simple_model):
    # Use model fixture from _models.py
    data_tga = total_gradient_amplitude_grid(simple_model)
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
