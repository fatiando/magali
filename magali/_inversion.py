# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Classes for inversions.
"""

import choclo
import numpy as np
import verde as vd
import verde.base as vdb
from scipy.optimize import minimize

from ._constants import MICROMETER_TO_METER
from ._synthetic import dipole_bz


class MagneticMomentBz:
    r"""
    Estimate magnetic dipole moment vector from Bz measurements.

    Uses the Bz component of the magnetic field to fit a point dipole model,
    returning the dipole moment vector that best fits the data in a
    least-squares sense.

    Parameters
    ----------
    location : tuple of floats
        Coordinates (x, y, z) of the dipole location, in µm.

    Attributes
    ----------
    dipole_moment_ : 1d-array
        Estimated dipole moment vector (mx, my, mz).
    jacobian : 2d-array
        Jacobian matrix evaluated at the data coordinates.

    Methods
    -------
    fit(coordinates, data)
        Fit the dipole model to the Bz component of the magnetic field.

    Notes
    -----
    The input/output magnetic field is assumed to be Bz in nT, and all
    coordinates should be provided in µm. Conversion to SI units is
    handled internally.
    """

    def __init__(self, location):
        self.location = location
        self.dipole_moment_ = None

    def _calculate_jacobian(self, x, y, z):
        """
        Compute the Jacobian matrix for the point dipole model.

        Parameters
        ----------
        x, y, z : 1d-arrays
            Coordinates of the observations, in micrometers.

        Returns
        -------
        jacobian : 2d-array
            Jacobian matrix (n_observations x 3).
        """
        x_c, y_c, z_c = vdb.n_1d_arrays(self.location, 3)
        factor = choclo.constants.VACUUM_MAGNETIC_PERMEABILITY / (4 * np.pi)
        x = x * MICROMETER_TO_METER
        y = y * MICROMETER_TO_METER
        z = z * MICROMETER_TO_METER
        x_c = x_c * MICROMETER_TO_METER
        y_c = y_c * MICROMETER_TO_METER
        z_c = z_c * MICROMETER_TO_METER
        n_data = x.size
        n_params = 3
        jacobian = [np.zeros(n_data) for _ in range(n_params)]

        for i in range(n_data):
            r = choclo.utils.distance_cartesian(x[i], y[i], z[i], x_c, y_c, z_c)
            kernel_eu = choclo.point.kernel_eu(x[i], y[i], z[i], x_c, y_c, z_c, r)
            kernel_nu = choclo.point.kernel_nu(x[i], y[i], z[i], x_c, y_c, z_c, r)
            kernel_uu = choclo.point.kernel_uu(x[i], y[i], z[i], x_c, y_c, z_c, r)

            jacobian[0][i] = factor * kernel_eu
            jacobian[1][i] = factor * kernel_nu
            jacobian[2][i] = factor * kernel_uu

        return np.stack(jacobian, axis=-1)

    def fit(self, coordinates, data):
        """
        Fit the magnetic dipole model to Bz data.

        Parameters
        ----------
        coordinates : tuple of arrays
            Coordinates (x, y, z) of the observations.
        data : array
            Observed Bz component of the magnetic field (in nT).

        Returns
        -------
        self
            This estimator instance, updated with the estimated `dipole_moment_` vector.
        """
        coordinates, data, _ = vdb.check_fit_input(coordinates, data, weights=None)
        x, y, z = vdb.n_1d_arrays(coordinates, 3)
        data = np.ravel(np.asarray(data * 1e-9))

        jacobian = self._calculate_jacobian(x, y, z)
        hessian = jacobian.T @ jacobian
        right_hand_system = jacobian.T @ data
        estimate = np.linalg.solve(hessian, right_hand_system)
        self.dipole_moment_ = estimate
        return self


class NonLinearMagneticMomentBz:
    r"""
    Estimate dipole location and moment from Bz data using nonlinear optimization.

    Fits a magnetic point dipole model to Bz measurements by simultaneously estimating
    the dipole position and moment vector that minimize the squared misfit between
    observed and predicted fields.

    Parameters
    ----------
    observed_data : xarray.Dataset
        Gridded dataset containing the observed Bz field.
    initial_position : array-like
        Initial guess for the dipole location (x, y, z), in meters.
    initial_moment : array-like
        Initial guess for the dipole moment vector (mx, my, mz), in Am².
    background_field : array-like or None, optional
        Optional background field to subtract from the observed Bz values.
    optimization_method : str, optional
        Optimization method to use (default is "Nelder-Mead"). Any method accepted
        by `scipy.optimize.minimize` can be used.

    Attributes
    ----------
    optimized_position_ : ndarray
        Estimated position of the magnetic dipole (x, y, z), in meters.
    optimized_moment_ : ndarray
        Estimated magnetic moment vector (mx, my, mz), in Am².
    result_ : OptimizeResult
        Full result object returned by `scipy.optimize.minimize`.

    Methods
    -------
    fit()
        Run non-linear optimization to simultaneously estimate position and moment.
    """

    def __init__(
        self,
        observed_data,
        initial_position,
        initial_moment,
        background_field=None,
        optimization_method="Nelder-Mead",
    ):
        self.observed_data = observed_data
        self.background_field = (
            np.ravel(np.asarray(background_field))
            if background_field is not None
            else None
        )
        self.initial_position = np.asarray(initial_position)
        self.initial_moment = np.asarray(initial_moment)
        self.optimization_method = optimization_method
        self.optimized_position_ = None
        self.optimized_moment_ = None
        self.result_ = None

    def _normalize(self):
        """
        Normalize parameters using the magnitude of the initial estimates.

        Returns
        -------
        x0_norm : array
            Normalized position vector.
        m0_norm : array
            Normalized moment vector.
        """
        self._x_scale = np.maximum(
            np.abs(self.initial_position), 1e-12
        )  # 1e-12 is used to avoid division by zero
        self._m_scale = np.linalg.norm(self.initial_moment) or 1e-12

        x0_norm = self.initial_position / self._x_scale
        m0_norm = self.initial_moment / self._m_scale
        return x0_norm, m0_norm

    def _denormalize(self, x_norm, m_norm):
        """
        Restore original scale from normalized parameters.

        Parameters
        ----------
        x_norm : array
            Normalized position vector.
        m_norm : array
            Normalized magnetic moment vector.

        Returns
        -------
        x : array
            Denormalized position vector.
        m : array
            Denormalized moment vector.
        """
        x = x_norm * self._x_scale
        m = m_norm * self._m_scale
        return x, m

    def _misfit(self, params):
        """
        Compute L2 norm misfit between base-corrected observed and predicted data.

        Misfit function: ||(d0 - b) - d(x, m)||^2

        Parameters
        ----------
        params : ndarray
            Flattened parameter vector
            [x_norm, y_norm, z_norm, mx_norm, my_norm, mz_norm].

        Returns
        -------
        misfit : float
            Sum of squared residuals between observed and predicted Bz values.
        """
        x = params[:3]
        m = params[3:]
        x, m = self._denormalize(x, m)
        data = vd.grid_to_table(self.observed_data)
        coordinates = (
            data.x,
            data.y,
            data.z,
        )
        model = dipole_bz(coordinates, x, m)
        if self.background_field is None:
            corrected_data = data.bz
        else:
            corrected_data = data.bz - self.background_field
        residuals = corrected_data - model
        return np.sum(residuals**2)

    def fit(self):
        """
        Estimate the dipole position and moment via nonlinear least-squares optimization.

        Uses the selected optimization method to minimize the misfit between the
        observed and predicted Bz values.

        Returns
        -------
        self : NonLinearMagneticMomentBz
            The instance, updated with the fitted `optimized_position_` and `optimized_moment_`.
        """
        x0_norm, m0_norm = self._normalize()
        initial_params = np.hstack((x0_norm, m0_norm))

        result = minimize(
            fun=self._misfit,
            x0=initial_params,
            method=self.optimization_method,
            options={"maxiter": 1000, "disp": True},
        )

        self.result_ = result
        x_norm_opt = result.x[:3]
        m_norm_opt = result.x[3:]
        self.optimized_position_, self.optimized_moment_ = self._denormalize(
            x_norm_opt, m_norm_opt
        )
        return self
