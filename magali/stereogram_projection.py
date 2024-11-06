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

rcparams = mpl.rcParams


class Stereoplot(LambertAxes):
    """
    A custom class for creating stereogram projections, particularly useful
    for applications in magnetic microscopy studies.

    The `Stereoplot` class extends the `LambertAxes` base class to implement
    a stereographic projection of vector orientations, such as magnetic field
    vectors. This projection is based on the `mplstereonet` project
    (https://github.com/joferkington/mplstereonet), which provides robust
    implementations for geological stereographic plots. This type of projection
    is valuable for visualizing orientation data in geoscience and related
    fields.

    """

    name = "stereoplot"
    _default_center_latitude = 0
    _default_center_longitude = 0

    def __init__(
        self, *args, center_latitude=None, center_longitude=None, rotation=0, **kwargs
    ):
        """
        Initialize the custom Axes object, similar to a standard Axes
        initialization, but with additional parameters for stereonet
        configuration.

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

        # Initialize overlay axes (for potential future use)
        self._overlay_axes = None

        super().__init__(*args, **kwargs)

    def calculate_stereonet_projection(self, azimuth, inclination):
        """
        Converts azimuth and inclination to x, y coordinates on a stereonet.

        Parameters
        ----------
        azimuth : float or array-like
            Azimuth angle(s) in degrees, measured clockwise from North.
        inclination : float or array-like
            Inclination angle(s) in degrees, with positive values downward and
            negative values upward.

        Returns
        -------
        x : float or array-like
            x-coordinate(s) on the stereonet.
        y : float or array-like
            y-coordinate(s) on the stereonet.

        Notes
        -----
        This function uses a stereographic projection, commonly used in
        geological and geophysical applications, where azimuth and inclination
        are mapped onto a plane for visual analysis of vector orientations.

        """

        azimuth_rad = np.radians(azimuth)
        inclination_rad = np.radians(inclination)

        # Compute the stereonet projection (Schmidt or Wulff projection)
        r = np.tan((np.pi / 4) - (inclination_rad / 2))
        x = r * np.sin(azimuth_rad)
        y = r * np.cos(azimuth_rad)

        return x, y


register_projection(Stereoplot)
