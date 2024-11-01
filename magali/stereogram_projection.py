# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#

import matplotlib as mpl
import numpy as np
from matplotlib.projections import register_projection
from matplotlib.projections.geo import LambertAxes

rcParams = mpl.rcParams


class Stereoplot(LambertAxes):
    """
    A custom class for stereogram projections applied to magnetic microscopy studies.
    """

    name = "stereoplot"
    _default_center_latitude = 0
    _default_center_longitude = 0
    _default_resolution = 60

    def __init__(
        self, *args, center_latitude=None, center_longitude=None, rotation=0, **kwargs
    ):
        """
        Initialize the custom Axes object, similar to a standard Axes
        initialization, but with additional parameters for stereonet configuration.

        Parameters
        ----------
        center_latitude : float, optional
            Center latitude of the stereonet in degrees
            (default is _default_center_latitude).
        center_longitude : float, optional
            Center longitude of the stereonet in degrees
            (default is _default_center_longitude).
        rotation : float, optional
            Rotation angle of the stereonet in degrees clockwise from North
            (default is 0).
        """
        # Store the rotation as radians (converted to a negative value)
        self.horizon = np.radians(90)
        self._rotation = -np.radians(rotation)

        # Set center latitude and longitude, using defaults if not provided
        center_latitude = center_latitude or self._default_center_latitude
        center_longitude = center_longitude or self._default_center_longitude

        # Store center latitude and longitude in kwargs for the base class
        kwargs["center_latitude"] = center_latitude
        kwargs["center_longitude"] = center_longitude
        kwargs.setdefault("resolution", self._default_resolution)

        # Initialize overlay axes (for potential future use)
        self._overlay_axes = None

        super().__init__(*args, **kwargs)


register_projection(Stereoplot)
