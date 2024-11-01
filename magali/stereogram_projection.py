# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#

import matplotlib as mpl
import numpy as np
from matplotlib.projections.geo import LambertAxes
from matplotlib.projections import register_projection

rcParams = mpl.rcParams


class Stereoplot(LambertAxes):
    """
    A custom class stereogram projections applied to magnetic microscopy
    studies.

    """

    name = "stereoplot"

    def __init__(self, *args, **kwargs):
        """
        Initialize the custom Axes object, similar to a standard Axes
        initialization,but with additional parameters for stereonet
        configuration.

        Parameters
        ----------
        center_lat : float, optional
            Center latitude of the stereonet in degrees
            (default is _default_center_lat).
        center_lon : float, optional
            Center longitude of the stereonet in degrees
            (default is _default_center_lon).
        rotation : float, optional
            Rotation angle of the stereonet in degrees clockwise from North
            (default is 0).
        """
        # Convert rotation to radians and store as a negative value
        self.horizon = np.radians(90)
        self._rotation = -np.radians(kwargs.pop("rotation", 0))

        # Extract center latitude and longitude, using defaults if not provided
        kwargs.setdefault("center_lat", self._default_center_lat)
        kwargs.setdefault("center_lon", self._default_center_lon)
        kwargs.setdefault("resolution", self._default_resolution)

        # Initialize overlay axes
        self._overlay_axes = None

        # Call LambertAxes base class constructor
        LambertAxes.__init__(*args, **kwargs)

register_projection(Stereoplot)