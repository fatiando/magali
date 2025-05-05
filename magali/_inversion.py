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
import verde.base as vdb

from ._constants import MICROMETER_TO_METER


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
        self.jacobian = None

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
        self.jacobian = jacobian
        hessian = jacobian.T @ jacobian
        hessian_inv = np.linalg.inv(hessian)
        estimate = hessian_inv @ jacobian.T @ data
        self.dipole_moment_ = estimate
        return self
