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
import numba
import numpy as np
from scipy.optimize import minimize

from ._synthetic import dipole_bz
from ._units import coordinates_micrometer_to_meter, tesla_to_nanotesla
from ._validation import check_fit_input


class MagneticMomentBz:
    """
    Estimate the magnetic dipole moment vector from Bz measurements.

    Uses the Bz component of the magnetic field to fit a point dipole model
    through a linear inversion [Souza-Junior2024]_, returning the dipole moment
    vector that best fits the data in a least-squares sense. Requires prior
    knowledge of the dipole location.

    Parameters
    ----------
    location : tuple = (float, float, float)
        Coordinates (x, y, z) of the dipole location, in µm.

    Attributes
    ----------
    dipole_moment_ : 1d-array
        Estimated dipole moment vector (mx, my, mz) in A.m². Only available
        after :meth:`~magali.MagneticMomentBz.fit` is called.

    References
    ----------
    [Souza-Junior2024]_
    """

    def __init__(self, location):
        self.location = location
        self.dipole_moment_ = None

    def fit(self, coordinates, data):
        """
        Fit the magnetic dipole model to Bz data.

        Parameters
        ----------
        coordinates : tuple = (x, y, z)
            Arrays with the x, y, and z coordinates of the observations points.
            The arrays can have any shape as long as they all have the same
            shape.
        data : array
            Array with the observed Bz component of the magnetic field (in nT)
            at the locations provided in *coordinates*. Must have the same
            shape as the coordinate arrays.

        Returns
        -------
        self
            This estimator instance, updated with the estimated dipole moment
            vector in the ``dipole_moment_`` attribute.
        """
        coordinates, data = check_fit_input(coordinates, data)
        jacobian = self.jacobian(coordinates)
        self.dipole_moment_ = np.linalg.solve(jacobian.T @ jacobian, jacobian.T @ data)
        return self

    def predict(self, coordinates):
        """
        Predict Bz values at given coordinates using the fitted dipole moment.

        Parameters
        ----------
        coordinates : tuple = (x, y, z)
            Arrays with the x, y, and z coordinates of the observations points.
            The arrays can have any shape as long as they all have the same
            shape.

        Returns
        -------
        ndarray
            Predicted Bz values at the specified coordinates.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.dipole_moment_ is None:
            msg = "Model has not been fitted. Call 'fit()' before prediction."
            raise ValueError(msg)

        return dipole_bz(
            coordinates,
            self.location,
            self.dipole_moment_,
        )

    def jacobian(self, coordinates):
        """
        Compute the Jacobian matrix for the linear point dipole model.

        The Jacobian is a matrix with derivatives of the forward modeling
        function (the magnetic field of a dipole) with regard to the parameters
        (the 3 dipole moment components) for each data point.

        Parameters
        ----------
        coordinates : tuple = (x, y, z)
            Arrays with the x, y, and z coordinates of the observations points.
            The arrays can have any shape as long as they all have the same
            shape.

        Returns
        -------
        jacobian : 2d-array
            The N x 3 Jacobian matrix, with N being the number of observations,
            in nT/(A.m²) units.
        """
        xc, yc, zc = coordinates_micrometer_to_meter(self.location)
        x, y, z = (c.ravel() for c in coordinates_micrometer_to_meter(coordinates))
        n_data = x.size
        n_params = 3
        jacobian = np.empty((n_data, n_params))
        jacobian_jit(x, y, z, xc, yc, zc, jacobian)
        jacobian = tesla_to_nanotesla(jacobian)
        return jacobian


def _jacobian(x, y, z, xc, yc, zc, result):
    """
    Jit-compiled version of the Jacobian matrix calculation.
    """
    factor = choclo.constants.VACUUM_MAGNETIC_PERMEABILITY / (4 * np.pi)
    n_data = x.size
    for i in numba.prange(n_data):
        r = choclo.utils.distance_cartesian(x[i], y[i], z[i], xc, yc, zc)
        kernel_eu = choclo.point.kernel_eu(x[i], y[i], z[i], xc, yc, zc, r)
        kernel_nu = choclo.point.kernel_nu(x[i], y[i], z[i], xc, yc, zc, r)
        kernel_uu = choclo.point.kernel_uu(x[i], y[i], z[i], xc, yc, zc, r)
        result[i, 0] = factor * kernel_eu
        result[i, 1] = factor * kernel_nu
        result[i, 2] = factor * kernel_uu


# Compile the Jacobian calculation. Doesn't use this as a decorator so that we
# can test the pure Python function and get coverage information about it.
jacobian_jit = numba.jit(_jacobian, nopython=True, parallel=True)


class NonlinearMagneticMomentBz:
    r"""
    Estimate dipole location and moment from Bz data using nonlinear optimization.

    Fits a magnetic point dipole model to Bz measurements by simultaneously
    estimating the dipole position and moment vector that minimize the squared
    misfit between observed and predicted fields [Souza-Junior2024]_.

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
        Optimization method to use (default is "Nelder-Mead"). Any method
        accepted by `scipy.optimize.minimize` can be used.

    Attributes
    ----------
    optimized_position_ : ndarray
        Estimated position of the magnetic dipole (x, y, z), in meters.
    optimized_moment_ : ndarray
        Estimated magnetic moment vector (mx, my, mz), in Am².
    result_ : OptimizeResult
        Full result object returned by `scipy.optimize.minimize`.

    References
    ----------
    [Souza-Junior2024]_
    """

    def __init__(
        self,
        initial_position,
        initial_moment,
        optimization_method="Nelder-Mead",
    ):
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
        self._x_scale = self.initial_position
        
        self._m_scale = np.linalg.norm(self.initial_moment)

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
        x_norm, m_norm = params[:3], params[3:]
        position, moment = self._denormalize(x_norm, m_norm)
        predicted = dipole_bz(self._coordinates, position, moment)
        residuals = self._data - predicted
        return np.linalg.norm(residuals)

    def fit(self, coordinates, data):
        """
        Estimate dipole position and moment using nonlinear least-squares.

        Uses the selected optimization method to minimize the misfit between
        the observed and predicted Bz values.

        Note:
            Any background field present in the system should be removed
            from the data prior to use.

        Parameters
        ----------
        coordinates : tuple = (x, y, z)
            Arrays with the x, y, and z coordinates of the observations points.
            The arrays can have any shape as long as they all have the same
            shape.
        data : array
            Array with the observed Bz component of the magnetic field (in nT)
            at the locations provided in *coordinates*. Must have the same
            shape as the coordinate arrays.

        Returns
        -------
        self : NonlinearMagneticMomentBz
            The instance, updated with the fitted `optimized_position_` and
            `optimized_moment_`.
        """
        coordinates, data = check_fit_input(coordinates, data)
        self._coordinates = coordinates
        self._data = data

        x0, m0 = self._normalize()
        initial_params = np.hstack((x0, m0))

        result = minimize(
            fun=self._misfit,
            x0=initial_params,
            method=self.optimization_method,
            options={"maxiter": 1000, "disp": True, "fatol":1e-8},
        )
        self.result_ = result
        x_opt, m_opt = result.x[:3], result.x[3:]
        self.position_, self.moment_ = self._denormalize(
            x_opt, m_opt
        )
        return self

    def predict(self, coordinates):
        """
        Predict Bz values at given coordinates using the fitted dipole parameters.

        Parameters
        ----------
        coordinates : tuple = (x, y, z)
            Arrays with the x, y, and z coordinates of the observations points.
            The arrays can have any shape as long as they all have the same
            shape.

        Returns
        -------
        ndarray
            Predicted Bz values at the specified coordinates.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.position_ is None or self.moment_ is None:
            msg = "Model has not been fitted. Call 'fit()' before prediction."
            raise ValueError(msg)

        return dipole_bz(
            coordinates,
            self.position_,
            self.moment_,
        )
