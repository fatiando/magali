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
import skimage.exposure
import verde as vd
import xarray as xr

from magali._detection import detect_anomalies
from magali._inversion import (
    MagneticMomentBz,
    NonlinearMagneticDipoleBz,
    _jacobian_linear,
    _jacobian_nonlinear,
    iterative_nonlinear_inversion,
)
from magali._synthetic import dipole_bz, dipole_bz_grid, random_directions
from magali._units import (
    coordinates_micrometer_to_meter,
    meter_to_micrometer,
    tesla_to_nanotesla,
)
from magali._utils import gradient
from magali._validation import check_fit_input


@pytest.fixture
def synthetic_data():
    SEED = 42
    rng = np.random.default_rng(SEED)

    sensor_sample_distance = 5.0
    region = [0, 2000, 0, 2000]
    spacing = 2.0
    true_inclination = 30
    true_declination = 40
    true_dispersion_angle = 5
    size = 5

    directions_inclination, directions_declination = random_directions(
        true_inclination,
        true_declination,
        true_dispersion_angle,
        size=size,
        random_state=SEED,
    )

    dipoles_amplitude = abs(rng.normal(0, 100, size)) * 1.0e-14
    dipole_coordinates = (
        rng.integers(30, 1970, size),
        rng.integers(30, 1970, size),
        rng.integers(-20, -1, size),
    )

    dipole_moments = hm.magnetic_angles_to_vec(
        inclination=directions_inclination,
        declination=directions_declination,
        intensity=dipoles_amplitude,
    )

    data = dipole_bz_grid(
        region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moments
    )
    noise_std_dev = 100
    data.values += rng.normal(loc=0, scale=noise_std_dev, size=data.shape)

    return data


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
    coordinates = tuple(c.ravel() for c in coordinates_micrometer_to_meter(coordinates))
    dipole_coordinates = tuple(
        c for c in coordinates_micrometer_to_meter(dipole_coordinates)
    )
    jacobian = np.empty((coordinates[0].size, 3))
    _jacobian_linear(*coordinates, *dipole_coordinates, jacobian)
    data_predicted = tesla_to_nanotesla(jacobian @ true_moment)
    np.testing.assert_allclose(data_predicted, data)


def test_nonlinear_magnetic_dipole_bz_inversion():
    "Check that the nonlinear inversion recovers a known dipole location and moment."
    coordinates = vd.grid_coordinates(
        region=[-20, 15, 0, 40],
        spacing=1,
        extra_coords=5,
    )
    true_location = (0, 20, -5)
    true_inclination = 45
    true_declination = -15
    true_intensity = 1e-15
    true_moment = hm.magnetic_angles_to_vec(
        inclination=true_inclination,
        declination=true_declination,
        intensity=true_intensity,
    )
    data = dipole_bz(coordinates, true_location, true_moment)

    initial_guess = (0.04041674, 20.27141785, 3.27582947)
    model = NonlinearMagneticDipoleBz(initial_location=initial_guess)
    # Check uninitialized attributes
    assert not hasattr(model, "location_")
    assert not hasattr(model, "dipole_moment_")
    assert not hasattr(model, "r2")

    # Run the inversion
    model.fit(coordinates, data)

    # Check that attributes are now set
    assert model.location_ is not None
    assert model.dipole_moment_ is not None
    assert model.r2_ is not None

    # Assert that results are close to the truth
    np.testing.assert_allclose(model.location_, true_location, atol=1.0)
    np.testing.assert_allclose(model.dipole_moment_, true_moment, rtol=0.05)
    np.testing.assert_allclose(model.r2_, 1, rtol=0.05)


def test_nonlinear_magnetic_dipole_jacobian_step_decreases_misfit():
    "Ensure the first LM step using the Jacobian decreases the residual norm."
    coordinates = vd.grid_coordinates(
        region=[-20, 15, 0, 40],
        spacing=1,
        extra_coords=5,
    )
    true_location = (0, 20, -5)
    true_inclination = 45
    true_declination = -15
    true_intensity = 1e-15
    true_moment = hm.magnetic_angles_to_vec(
        inclination=true_inclination,
        declination=true_declination,
        intensity=true_intensity,
    )
    data = dipole_bz(coordinates, true_location, true_moment)

    initial_guess = (0.04041674, 20.27141785, 3.27582947)
    model = NonlinearMagneticDipoleBz(initial_location=initial_guess, max_iter=1)

    model.fit(coordinates, data)

    # Should have completed 1 outer iteration, hence 2 misfit values
    assert len(model.misfit_) == 2
    assert model.misfit_[1] < model.misfit_[0]


def test_nonlinear_magnetic_dipole_predict():
    "Check that predict returns accurate Bz values after fitting."
    coordinates = vd.grid_coordinates(
        region=[-20, 15, 0, 40],
        spacing=1,
        extra_coords=5,
    )
    true_location = (0, 20, -5)
    true_inclination = 45
    true_declination = -15
    true_intensity = 1e-15
    true_moment = hm.magnetic_angles_to_vec(
        inclination=true_inclination,
        declination=true_declination,
        intensity=true_intensity,
    )
    data = dipole_bz(coordinates, true_location, true_moment)

    # Fit the model
    model = NonlinearMagneticDipoleBz(
        initial_location=(0.04041674, 20.27141785, 3.27582947)
    )
    model.fit(coordinates, data)

    # Predict and compare
    predicted = model.predict(coordinates)
    assert predicted.shape == data.shape
    np.testing.assert_allclose(predicted, data, rtol=1e-2, atol=1.0)

    # Check error if predict is called before fit
    model_unfit = NonlinearMagneticDipoleBz(initial_location=(450, 450, -10))
    with pytest.raises(AttributeError):
        model_unfit.predict(coordinates)


def test_nonlinear_magnetic_moment_gz_jacobian():
    "Make sure the non-jitted nonlinear Jacobian calculation is correct"
    coordinates = vd.grid_coordinates(
        region=[-20, 15, 0, 40],
        spacing=1,
        extra_coords=5,
    )
    dipole_coordinates = (0, 20, -5)
    true_inclination = 45
    true_declination = -15
    true_intensity = 1e-15
    true_moment = hm.magnetic_angles_to_vec(
        inclination=true_inclination,
        declination=true_declination,
        intensity=true_intensity,
    )
    data = dipole_bz(coordinates, dipole_coordinates, true_moment)

    coordinates, data = check_fit_input(coordinates, data)
    dipole_coordinates = tuple(
        c.ravel() for c in coordinates_micrometer_to_meter(coordinates)
    )
    location = np.asarray((0.04041674, 20.27141785, 3.27582947))
    linear_model = MagneticMomentBz(location)
    linear_model.fit(coordinates, data)
    moment = linear_model.dipole_moment_
    residual = data - dipole_bz(coordinates, location, moment)
    jacobian = np.empty((data.size, 3))
    _jacobian_nonlinear(*coordinates, *location, *moment, jacobian)
    hessian = jacobian.T @ jacobian
    gradient = jacobian.T @ residual
    alpha = 1
    identity = np.identity(3)

    delta = np.linalg.solve(hessian + alpha * identity, gradient)
    trial_location = location + meter_to_micrometer(delta)
    predicted = dipole_bz(
        coordinates,
        trial_location,
        moment,
    )
    residual = data - predicted
    np.testing.assert_allclose(max(residual), 80.94042377738671, atol=1e-3)
    np.testing.assert_allclose(min(residual), -149.98554665789848, atol=1e-3)
    np.testing.assert_allclose(np.mean(residual), -10.092100833939794, atol=1e-3)
    np.testing.assert_allclose(np.std(residual), 31.744971328845484, atol=1e-3)


def test_iterative_nonlinear_inversion(synthetic_data):
    height_difference = 5.0
    data_up = (
        hm.upward_continuation(synthetic_data, height_difference)
        .assign_attrs(synthetic_data.attrs)
        .assign_coords(x=synthetic_data.x, y=synthetic_data.y)
        .assign_coords(z=synthetic_data.z + height_difference)
        .rename("bz")
    )
    dx, dy, dz, tga = gradient(data_up)
    data_up["dx"], data_up["dy"], data_up["dz"], data_up["tga"] = dx, dy, dz, tga
    stretched = skimage.exposure.rescale_intensity(
        tga, in_range=tuple(np.percentile(tga, (1, 99)))
    )
    data_tga_stretched = xr.DataArray(stretched, coords=data_up.coords)
    bounding_boxes = detect_anomalies(
        data_tga_stretched,
        size_range=[30, 50],
        detection_threshold=0.03,
        border_exclusion=2,
        size_multiplier=1,
    )
    data_updated, locations_, dipole_moments_, r2_values = (
        iterative_nonlinear_inversion(
            data_up,
            bounding_boxes,
            height_difference=height_difference,
            copy_data=True,
        )
    )
    assert isinstance(data_updated, xr.DataArray)
    assert isinstance(locations_, list)
    assert isinstance(dipole_moments_, list)
    assert isinstance(r2_values, list)
    assert len(locations_) == len(dipole_moments_) == len(r2_values)
    np.testing.assert_allclose(locations_[0][0], 1506.00100449, atol=1e-3)
    np.testing.assert_allclose(dipole_moments_[0][0], 5.22732937e-13, atol=1e-12)


def test_nonlinear_inner_loop_no_step_taken():
    """
    Test the scenario where inner loop fails to find a step that reduces misfit.
    This covers the 'if not took_a_step: break' condition.
    """
    coordinates = vd.grid_coordinates(
        region=[-10, 10, -10, 10],
        spacing=2,
        extra_coords=5,
    )
    true_location = (0, 0, -5)
    true_moment = np.array([1e-12, 0, 1e-12])

    data = dipole_bz(coordinates, true_location, true_moment)

    initial_location = np.array([1000.0, 1000.0, 1000.0])  # Very far from solution

    model = NonlinearMagneticDipoleBz(
        initial_location=initial_location,
        max_iter=3,  # Few outer iterations
        tol=1e-10,  # Very tight tolerance
        alpha_init=1e20,  # Extremely high damping
        alpha_scale=1.001,  # Very slow reduction
    )

    model.fit(coordinates, data)

    assert hasattr(model, "location_")
    assert hasattr(model, "dipole_moment_")
    assert hasattr(model, "misfit_")
    assert len(model.misfit_) >= 2


def test_nonlinear_inner_loop_tolerance_convergence():
    """
    Test inner loop convergence via tolerance condition rather than step success.
    Covers the inner loop tolerance break condition.
    """
    coordinates = vd.grid_coordinates(
        region=[-5, 5, -5, 5],
        spacing=1,
        extra_coords=2,
    )
    true_location = (0.0, 0.0, -3.0)
    true_moment = np.array([1e-12, 1e-12, 1e-12])

    data = dipole_bz(coordinates, true_location, true_moment)

    # Very close to true location
    initial_location = np.array([0.1, 0.1, -2.9])

    model = NonlinearMagneticDipoleBz(
        initial_location=initial_location,
        max_iter=10,
        tol=0.5,  # Very loose tolerance
        alpha_init=1.0,
        alpha_scale=10.0,
    )

    model.fit(coordinates, data)

    assert model.misfit_[-1] < model.misfit_[0]
    assert len(model.misfit_) > 1


def test_nonlinear_max_step_normalization():
    """
    Test that step normalization works when step norm exceeds max_step_m.
    This covers the step_norm > max_step_m condition.
    """
    coordinates = vd.grid_coordinates(
        region=[-20, 20, -20, 20],
        spacing=5,
        extra_coords=10,
    )
    true_location = (0.0, 0.0, -8.0)
    true_moment = np.array([5e-11, 5e-11, 5e-11])  # Large moment for large gradients

    data = dipole_bz(coordinates, true_location, true_moment)

    initial_location = np.array([50.0, 50.0, 50.0])

    model = NonlinearMagneticDipoleBz(
        initial_location=initial_location,
        max_iter=10,
        tol=1e-2,
        alpha_init=0.01,  # Low damping for larger steps
        alpha_scale=5.0,
    )

    model.fit(coordinates, data)

    assert model.misfit_[-1] < model.misfit_[0]
    assert not np.allclose(model.location_, initial_location)


def test_nonlinear_outer_loop_tolerance_convergence():
    """
    Test outer loop convergence via tolerance condition.
    This ensures the outer loop break condition is covered.
    """
    coordinates = vd.grid_coordinates(
        region=[-10, 10, -10, 10],
        spacing=2,
        extra_coords=3,
    )
    true_location = (1.0, -1.0, -4.0)
    true_moment = np.array([2e-12, -1e-12, 2e-12])
    
    data = dipole_bz(coordinates, true_location, true_moment)
    
    initial_location = np.array([1.1, -0.9, -3.9])
    
    model = NonlinearMagneticDipoleBz(
        initial_location=initial_location,
        max_iter=100,
        tol=0.1,  # Loose tolerance for quick outer loop convergence
        alpha_init=1.0,
        alpha_scale=10.0
    )
    
    model.fit(coordinates, data)
    
    assert len(model.misfit_) >= 2
    assert model.r2_ > 0.9

def test_nonlinear_single_iteration():
    """
    Test behavior with only one iteration.
    This covers the case where outer loop runs only once.
    """
    coordinates = vd.grid_coordinates(
        region=[-5, 5, -5, 5],
        spacing=1,
        extra_coords=2,
    )
    true_location = np.array([0.0, 0.0, -2.0])
    true_moment = np.array([1e-12, 0, 1e-12])
    
    data = dipole_bz(coordinates, true_location, true_moment)
    
    # Use true location as initial guess (perfect scenario)
    initial_location = true_location.copy()
    
    model = NonlinearMagneticDipoleBz(
        initial_location=initial_location,
        max_iter=1,  # Only one iteration
        tol=1e-10,
        alpha_init=1.0,
        alpha_scale=10.0
    )
    
    model.fit(coordinates, data)
    
    assert hasattr(model, 'location_')
    assert hasattr(model, 'dipole_moment_')
    assert hasattr(model, 'misfit_')
    assert len(model.misfit_) == 2