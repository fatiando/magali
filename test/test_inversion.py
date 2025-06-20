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

from magali._inversion import MagneticMomentBz, _jacobian
from magali._synthetic import dipole_bz


def test_linear_magnetic_moment_bz_inversion():
    "Check that the inversion recovers a known direction."
    dipole_coordinates = (500, 500, -15)
    true_inclination = 30
    true_declination = 40
    true_intensity = 5e-11
    true_moment = hm.magnetic_angles_to_vec(
        inclination=true_inclination,
        declination=true_declination,
        intensity=true_intensity,
    )
    coordinates = vd.grid_coordinates(
        region=[0, 100, 0, 100],
        spacing=1,
        extra_coords=5,
    )
    data = dipole_bz(coordinates, dipole_coordinates, true_moment)
    model = MagneticMomentBz(dipole_coordinates)
    # Test initalization
    assert model.location == dipole_coordinates
    assert model.dipole_moment_ is None
    # Run the inversion
    model.fit(coordinates, data)
    # Assert estimated moment is close to true moment
    assert model.dipole_moment_ is not None
    np.testing.assert_allclose(model.dipole_moment_, true_moment)
    # This fails because of a bug in the Harmonica function. Uncomment once
    # 0.7.1 is out with a bug fix.
    # intensity, inclination, declination = hm.magnetic_vec_to_angles(*model.dipole_moment_)
    # np.testing.assert_allclose(intensity, true_intensity)
    # np.testing.assert_allclose(declination, true_declination)
    # np.testing.assert_allclose(inclination, true_inclination)


def test_linear_magnetic_moment_gz_jacobian():
    "Make sure the non-jitted Jacobian calculation is correct"
    dipole_coordinates = (500, 500, -15)
    true_inclination = 30
    true_declination = 40
    true_intensity = 5e-11
    true_moment = hm.magnetic_angles_to_vec(
        inclination=true_inclination,
        declination=true_declination,
        intensity=true_intensity,
    )
    coordinates = vd.grid_coordinates(
        region=[0, 100, 0, 100],
        spacing=2,
        extra_coords=5,
    )
    data = dipole_bz(coordinates, dipole_coordinates, true_moment).ravel()
    # Convert to meters for the Jacobian calculation
    coordinates = tuple(c.ravel() * 1e-6 for c in coordinates)
    dipole_coordinates = tuple(c * 1e-6 for c in dipole_coordinates)
    jacobian = np.empty((coordinates[0].size, 3))
    _jacobian(*coordinates, *dipole_coordinates, jacobian)
    data_predicted = jacobian @ true_moment * 1e9
    np.testing.assert_allclose(data_predicted, data)
