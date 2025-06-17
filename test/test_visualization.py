# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the visualization functions
"""

import harmonica as hm
import matplotlib.pyplot as plt
import numpy as np
import pytest
import skimage.exposure

from magali._detection import detect_anomalies
from magali._synthetic import (
    dipole_bz_grid,
    random_directions,
)
from magali._utils import total_gradient_amplitude_grid
from magali._visualization import plot_bounding_boxes


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plot_bounding_boxes(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    sensor_sample_distance = 5.0  # µm
    region = [0, 2000, 0, 2000]  # µm
    spacing = 2  # µm
    size = 100

    inc, dec = random_directions(30, 40, 5, size=size, random_state=5)
    amp = abs(np.random.normal(0, 100, size)) * 1.0e-14
    coords = (
        np.concatenate([np.random.randint(30, 1970, size), [1250, 1300, 500]]),
        np.concatenate([np.random.randint(30, 1970, size), [500, 1750, 1000]]),
        np.concatenate([np.random.randint(-20, -1, size), [-15, -15, -30]]),
    )
    moments = hm.magnetic_angles_to_vec(
        inclination=np.concatenate([inc, [10, -10, -5]]),
        declination=np.concatenate([dec, [10, 170, 190]]),
        intensity=np.concatenate([amp, [5e-11, 5e-11, 5e-11]]),
    )
    data = dipole_bz_grid(region, spacing, sensor_sample_distance, coords, moments)

    height_diff = 5
    data_up = (
        hm.upward_continuation(data, height_diff)
        .assign_attrs(data.attrs)
        .assign_coords(x=data.x, y=data.y)
        .assign_coords(z=data.z + height_diff)
    )
    tga = total_gradient_amplitude_grid(data_up)
    tga_stretched = skimage.exposure.rescale_intensity(
        tga, in_range=tuple(np.percentile(tga, (1, 99)))
    )

    windows = detect_anomalies(
        tga_stretched,
        size_range=[25, 50],
        size_multiplier=2,
        num_scales=10,
        detection_threshold=0.01,
        overlap_ratio=0.5,
        border_exclusion=1,
    )

    plot_bounding_boxes(tga_stretched, windows, title="Test Anomalies Plot")
