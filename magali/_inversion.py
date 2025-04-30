# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Classes for inversions.
"""

import numpy as np
import scipy as sp
import verde.base as vdb


class MagneticMomentBz:
    def __init__(self, location):
        self.location = location
        # The estimated parameters. Start them with None
        self.dipole_moment_ = None

    def fit(self, coordinates, data):
        "data - bz vector"
        coordinates, data, _ = vdb.check_fit_input(coordinates, data, weights=None)
        x, y, z = vdb.n_1d_arrays(coordinates, 3)
        r = np.sqrt(
            (coordinates[0] - self.location[0]) ** 2
            + (coordinates[1] - self.location[1]) ** 2
            + (coordinates[2] - self.location[2]) ** 2
        )
        r_5 = r**5
        
