# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the _detection functions
"""

from models import simple_model

from magali._detection import detect_anomalies


def test_detect_anomalies(simple_model):
    # Use model fixture from _models.py
    windows = detect_anomalies(
        simple_model,
        size_range=[25, 50],
        size_multiplier=2,
        num_scales=10,
        detection_threshold=0.01,
        overlap_ratio=0.5,
        border_exclusion=1,
    )

    assert len(windows) == 3
    assert len(windows[0]) == 4
