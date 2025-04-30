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
        x_c, y_c, z_c = vdb.n_1d_arrays(self.location, 3)
        r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2 + (z - z_c) ** 2)
        r_5 = r**5
        n_data = data.size
        jacobian = np.empty((n_data, 3))
