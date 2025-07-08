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

from ._synthetic import dipole_bz
from ._units import (
    coordinates_micrometer_to_meter,
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


def _jacobian_kernel(x, y, z, xc, yc, zc, mx, my, mz, result):
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

        result[i, 0] = dBz_dxc
        result[i, 1] = dBz_dyc
        result[i, 2] = dBz_dzc


jacobian_kernel_jit = numba.jit(_jacobian_kernel, nopython=True, parallel=True)


class NonlinearMagneticDipoleBz:
    def __init__(self, location, dipole_moment):
        self.location = location
        self.dipole_moment = dipole_moment
        self.location_ = None
        self.moment_ = None
        self.parameters_ = None
        self.jacobian = None

        self.scale_location = 1e-6
        self.scale_moment = 1e-10

    def fit(
        self,
        coordinates,
        data,
        max_iter=100,
        tol=1e-10,
        lambda_init=1e-2,
        lambda_scale=10.0,
    ):
        coordinates, data = check_fit_input(coordinates, data)
        misfit = []

        lambda_ = lambda_init
        position = self.location
        for _ in range(max_iter):
            linear_model = MagneticMomentBz(position)
            linear_model.fit(coordinates, data)
            moment = linear_model.dipole_moment_

            predicted_data = dipole_bz(coordinates, position, moment)
            residual = data - predicted_data
            current_misfit = np.linalg.norm(residual) ** 2
            misfit.append(current_misfit)

            A = np.empty((data.size, 3))

            x, y, z = coordinates_micrometer_to_meter(coordinates)
            x = x.ravel()
            y = y.ravel()
            z = z.ravel()

            xc, yc, zc = coordinates_micrometer_to_meter(self.location)

            jacobian_kernel_jit(
                x,
                y,
                z,
                xc,
                yc,
                zc,
                moment[0],
                moment[1],
                moment[2],
                A,
            )

            JTJ = A.T @ A
            JTr = A.T @ (residual)

            accepted = False

            for _ in range(50):
                delta = np.linalg.solve(JTJ + lambda_ * np.eye(3), JTr)
                trial_position = position + delta

                linear_model = MagneticMomentBz(trial_position)
                linear_model.fit(coordinates, data)
                trial_moment = linear_model.dipole_moment_
                trial_pred = dipole_bz(coordinates, trial_position, trial_moment)
                trial_residual = data - trial_pred
                trial_misfit = np.linalg.norm(trial_residual) ** 2

                if trial_misfit < current_misfit:
                    position = trial_position
                    lambda_ /= lambda_scale
                    misfit.append(trial_misfit)
                    accepted = True
                    break
                lambda_ *= lambda_scale

            if not accepted:
                break

            if abs(misfit[-1] - misfit[-2]) / misfit[-2] < tol:
                break

        self.location_ = position
        self.moment_ = moment
        return position, moment


# Compile the Jacobian calculation. Doesn't use this as a decorator so that we
# can test the pure Python function and get coverage information about it.
jacobian_jit = numba.jit(_jacobian, nopython=True, parallel=True)
