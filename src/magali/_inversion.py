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
from scipy.linalg import solve

from ._synthetic import dipole_bz
from ._units import (
    coordinates_micrometer_to_meter,
    meter_to_micrometer,
    tesla_to_nanotesla,
)
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
        jacobian_linear_jit(x, y, z, xc, yc, zc, jacobian)
        jacobian = tesla_to_nanotesla(jacobian)
        return jacobian


def _jacobian_linear(x, y, z, xc, yc, zc, result):
    """
    Jit-compiled version of the Jacobian matrix calculation for the linear inversion.
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


class NonlinearMagneticDipoleBz:
    """
    Estimate the position and magnetic dipole moment vector from Bz measurements.

    Uses the Bz component of the magnetic field to estimate both the position
    and the moment of a magnetic dipole through a nonlinear inversion based on
    the Levenberg-Marquardt algorithm. Returns the location and dipole moment
    vector that best fit the data in a least-squares sense. Requires an initial
    guess for the dipole location.

    Parameters
    ----------
    initial_location : tuple of float
        Initial guess for the coordinates (x, y, z) of the dipole location, in µm.
    max_iter : int
        Maximum number of iterations for both the outer and inner loops of the
        nonlinear inversion.
    tol : float
        Convergence tolerance for the relative change in misfit between iterations.
    alpha_init : float
        Initial damping parameter for the Levenberg-Marquardt algorithm.
    alpha_scale : float
        Multiplicative factor used to increase or decrease the damping parameter
        during the optimization.

    Attributes
    ----------
    location_ : 1d-array
        Estimated location (x, y, z) of the dipole, in µm.
    dipole_moment_ : 1d-array
        Estimated dipole moment vector (mx, my, mz), in A.m².
    misfit_ : list of float
        Norm of the residual vector at each outer iteration of the inversion.

    References
    ----------
    [Souza-Junior2024]_
    """

    def __init__(
        self,
        initial_location,
        max_iter=100,
        tol=1e-4,
        alpha_init=1,
        alpha_scale=10.0,
    ):
        self.initial_location = initial_location
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_init = alpha_init
        self.alpha_scale = alpha_scale

    def predict(self, coordinates):
        """
        Predict the Bz magnetic field from the estimated dipole parameters.

        Uses the estimated dipole location and moment vector to compute the
        Bz component of the magnetic field at the given observation points.

        Parameters
        ----------
        coordinates : tuple of array-like
            Arrays with the x, y,    z coordinates of the observation points,
            in µm. The arrays must have the same shape.

        Returns
        -------
        predicted : array
            Array with the predicted Bz values (in nT) at the observation points.
            Has the same shape as the input coordinate arrays.

        Raises
        ------
        AttributeError
            If :meth:`~NonlinearMagneticDipoleBz.fit` has not been called yet,
            and ``location_`` or ``dipole_moment_`` are not set.

        See Also
        --------
        fit : Estimate the dipole location and moment from Bz measurements.
        """
        if not hasattr(self, "location_") or not hasattr(self, "dipole_moment_"):
            msg = "Model has not been fitted yet. Call 'fit' before 'predict'."
            raise AttributeError(msg)
        return dipole_bz(coordinates, self.location_, self.dipole_moment_)

    def jacobian(self, coordinates, location, moment, jacobian):
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
        location : array-like
            Dipole location (x, y, z), in µm.
        moment : array-like
            Dipole moment vector (mx, my, mz), in A.m².
        jacobian : (n, 3) ndarray
            Preallocated array to store the resulting Jacobian matrix,
            where n is the number of observation points.

        Returns
        -------
        jacobian : (n, 3) ndarray
            The Jacobian matrix of Bz with respect to the dipole moment,
            in nT/(A·m²). Each row corresponds to an observation point, and
            each column to the partial derivative with respect to mx, my,
            and mz, respectively.
        """
        x, y, z = coordinates
        xc, yc, zc = coordinates_micrometer_to_meter(location)
        jacobian_nonlinear_jit(
            x,
            y,
            z,
            xc,
            yc,
            zc,
            moment[0],
            moment[1],
            moment[2],
            jacobian,
        )
        return jacobian

    def fit(self, coordinates, data):
        r"""
        Fit the nonlinear magnetic dipole model to Bz data.

        Performs nonlinear inversion using the Levenberg-Marquardt method to
        estimate both the dipole location and its magnetic moment. The method
        alternates between a nonlinear update of the dipole location (inner loop)
        and a linear least-squares estimate of the dipole moment (outer loop).
        The Jacobian matrix with respect to the location is computed numerically
        using JIT-accelerated code.

        The Jacobian matrix used in the nonlinear step contains partial
        derivatives of the Bz field with respect to the dipole location:
        :math:`\frac{\partial B_z}{\partial x_0}`, :math:`\frac{\partial B_z}{\partial y_0}`,
        and :math:`\frac{\partial B_z}{\partial z_0}`. These are computed assuming a fixed
        moment vector.

        At each inner iteration:

        - The forward field is computed using the trial location and fixed moment.
        - A trial update is accepted if it reduces the data misfit.
        - The damping parameter (alpha) is adapted based on success/failure.

        At each outer iteration:

        - A new linear estimate of the dipole moment is computed for the current location.
        - Convergence is assessed based on relative reduction in residual norm.

        Parameters
        ----------
        coordinates : tuple of array-like
            Arrays with the x, y, z coordinates of the observation points.
            The arrays can have any shape as long as they all have the same shape.
        data : array-like
            Observed Bz component of the magnetic field (in nT) at the observation
            points. Must have the same shape as the coordinate arrays.

        Returns
        -------
        self : object
            This instance with updated ``location_`` and ``dipole_moment_``.

        Notes
        -----
        Internally uses:

        - :func:`jacobian_nonlinear_jit`: JIT-compiled function that fills the
          Jacobian matrix :math:`\frac{\partial B_z}{\partial (x_0, y_0, z_0)}`  for a fixed moment.
        - :func:`dipole_bz`: forward model for the Bz field of a dipole at given
          coordinates.
        - :class:`MagneticMomentBz`: linear inversion for estimating moment given a
          fixed location.
        """
        coordinates, data = check_fit_input(coordinates, data)
        coordinates_m = tuple(
            c.ravel() for c in coordinates_micrometer_to_meter(coordinates)
        )
        location = np.asarray(self.initial_location)
        linear_model = MagneticMomentBz(location)
        linear_model.fit(coordinates, data)
        moment = linear_model.dipole_moment_
        residual = data - dipole_bz(coordinates, location, moment)
        misfit = [np.linalg.norm(residual)]
        alpha = self.alpha_init
        jacobian = np.empty((data.size, 3))
        identity = np.identity(3)
        for _ in range(self.max_iter):
            location_misfit = [misfit[-1]]
            for _ in range(self.max_iter):
                jacobian = self.jacobian(coordinates_m, location, moment, jacobian)
                hessian = jacobian.T @ jacobian
                gradient = jacobian.T @ residual
                took_a_step = False
                for _ in range(50):
                    delta = np.linalg.solve(hessian + alpha * identity, gradient)
                    trial_location = location + meter_to_micrometer(delta)
                    trial_predicted = dipole_bz(
                        coordinates,
                        trial_location,
                        moment,
                    )
                    trial_residual = data - trial_predicted
                    trial_misfit = np.linalg.norm(trial_residual)
                    if trial_misfit < location_misfit[-1]:
                        location = trial_location
                        residual = trial_residual
                        alpha /= self.alpha_scale
                        location_misfit.append(trial_misfit)
                        took_a_step = True
                        break
                    alpha *= self.alpha_scale
                if not took_a_step:
                    break
                if (
                    abs(location_misfit[-1] - location_misfit[-2]) / location_misfit[-2]
                    < self.tol
                ):
                    break
            linear_model = MagneticMomentBz(location).fit(coordinates, data)
            moment = linear_model.dipole_moment_
            residual = data - dipole_bz(coordinates, location, moment)
            misfit.append(np.linalg.norm(residual))
            if abs(misfit[-1] - misfit[-2]) / misfit[-2] < self.tol:
                break
        self.location_ = location
        self.dipole_moment_ = moment
        self.misfit_ = misfit
        return self


def _jacobian_nonlinear(x, y, z, xc, yc, zc, mx, my, mz, result):
    """
    Jit-compiled version of the Jacobian matrix calculation for the nonlinear inversion.
    """
    factor = choclo.constants.VACUUM_MAGNETIC_PERMEABILITY / (4 * np.pi)
    for i in numba.prange(x.size):
        dx = x[i] - xc
        dy = y[i] - yc
        dz = z[i] - zc
        r2 = dx**2 + dy**2 + dz**2
        r5 = r2 ** (5 / 2)
        r7 = r2 ** (7 / 2)
        # ∂bz / ∂xc
        dBz_dxc = factor * (
            (15 * my * dy * dz * dx) / r7
            + mz * ((15 * dz**2 * dx) / r7 - (3 * dx) / r5)
            + (15 * mx * dz * dx**2) / r7
            - (3 * mx * dz) / r5
        )
        # ∂bz / ∂yc
        dBz_dyc = factor * (
            (15 * mx * dx * dz * dy) / r7
            + mz * ((15 * dz**2 * dy) / r7 - (3 * dy) / r5)
            + (15 * my * dz * dy**2) / r7
            - (3 * my * dz) / r5
        )
        # ∂bz / ∂zc
        dBz_dzc = factor * (
            mz * ((15 * dz**3) / r7 - (9 * dz) / r5)
            + (15 * my * dy * dz**2) / r7
            - (3 * my * dy) / r5
            + (15 * mx * dx * dz**2) / r7
            - (3 * mx * dx) / r5
        )
        # Convert to nT
        result[i, 0] = dBz_dxc * 1e9
        result[i, 1] = dBz_dyc * 1e9
        result[i, 2] = dBz_dzc * 1e9


# Compile the Jacobian calculation. Doesn't use this as a decorator so that we
# can test the pure Python function and get coverage information about it.
jacobian_linear_jit = numba.jit(_jacobian_linear, nopython=True, parallel=True)
jacobian_nonlinear_jit = numba.jit(_jacobian_nonlinear, nopython=True, parallel=True)
