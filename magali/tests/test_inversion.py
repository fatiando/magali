# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the _synthetic functions
"""

import harmonica as hm
import numpy as np
import verde as vd

from .._inversion import MagneticMomentBz
from .._synthetic import dipole_bz_grid, random_directions


def test_MagneticMomentBz():
    sensor_sample_distance = 5.0  # µm
    region = [0, 1000, 0, 1000]  # µm
    spacing = 2  # µm

    true_inclination = 30
    true_declination = 40
    true_dispersion_angle = 5

    coordinates = vd.grid_coordinates(
        region=region,  # µm
        spacing=spacing,  # µm
        extra_coords=sensor_sample_distance,
    )

    size = 1

    directions_inclination, directions_declination = random_directions(
        true_inclination,
        true_declination,
        true_dispersion_angle,
        size=size,
        random_state=5,
    )

    dipole_coordinates = (500, 500, -15)

    dipole_moment = hm.magnetic_angles_to_vec(
        inclination=directions_inclination,
        declination=directions_declination,
        intensity=5e-11,
    )

    data = dipole_bz_grid(
        region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moment
    )

    data.plot.pcolormesh(cmap="seismic", vmin=-5000, vmax=5000)

    model = MagneticMomentBz(dipole_coordinates)

    # Test initalization
    assert model.location == dipole_coordinates
    assert model.dipole_moment_ is None

    model.fit(coordinates, data)
    dipole_moment = np.array(
        [dipole_moment[0][0], dipole_moment[1][0], dipole_moment[2][0]]
    )

    # Assert estimated moment is close to true moment
    assert model.dipole_moment_ is not None
    np.testing.assert_allclose(model.dipole_moment_, dipole_moment, rtol=1e2)
