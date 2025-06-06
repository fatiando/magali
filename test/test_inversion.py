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
import pytest
import verde as vd

from magali._inversion import MagneticMomentBz, NonlinearMagneticMomentBz, _jacobian
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


def test_linear_magnetic_moment_bz_inversion_predict():
    "Check that linear inversion recovers a known position and moment."
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

    predicted = model.predict(coordinates)

    np.testing.assert_allclose(predicted, data, rtol=0.2)


def test_linear_predict_without_fit_raises():
    "Ensure ValueError is raised if predict is called before fit"
    dipole_coordinates = (500, 500, -15)

    coordinates = vd.grid_coordinates(
        region=[0, 100, 0, 100],
        spacing=1,
        extra_coords=5,
    )

    model = MagneticMomentBz(dipole_coordinates)
    coordinates = vd.grid_coordinates(
        region=[0, 100, 0, 100],
        spacing=10,
        extra_coords=5,
    )
    with pytest.raises(ValueError, match="Model has not been fitted"):
        model.predict(coordinates)


def test_linear_magnetic_moment_bz_jacobian():
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


def test_nonlinear_magnetic_moment_bz_inversion():
    "Check that nonlinear inversion recovers a known position and moment."
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

    nl_inv = NonlinearMagneticMomentBz(
        initial_position=[490, 480, -13],
        initial_moment=hm.magnetic_angles_to_vec(
        inclination=29,
        declination=39,
        intensity=1e-11,
    ),
    )

    nl_inv.fit(coordinates, data)

    # Check if estimated values are close to true ones
    np.testing.assert_allclose(nl_inv.position_, dipole_coordinates)
    np.testing.assert_allclose(nl_inv.moment_, true_moment)


def test_nonlinear_magnetic_moment_bz_inversion_predict():
    "Check that nonlinear inversion recovers a known position and moment."
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
    model.fit(coordinates, data)

    nl_inv = NonlinearMagneticMomentBz(
        initial_position=model.location,
        initial_moment=model.dipole_moment_,
    )

    nl_inv.fit(coordinates, data)

    predicted = nl_inv.predict(coordinates)

    np.testing.assert_allclose(predicted, data, rtol=0.2)


def test_nonlinear_predict_without_fit_raises():
    "Ensure ValueError is raised if predict is called before fit"
    model = NonlinearMagneticMomentBz(
        initial_position=(100, 100, -5),
        initial_moment=[1e-11, 1e-11, 1e-11],
    )
    coordinates = vd.grid_coordinates(
        region=[0, 100, 0, 100],
        spacing=10,
        extra_coords=5,
    )
    with pytest.raises(ValueError, match="Model has not been fitted"):
        model.predict(coordinates)
