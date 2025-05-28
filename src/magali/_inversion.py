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
        coordinates, data, _ = vdb.check_fit_input(coordinates, data, weights=None)
        # Convert the data from nT to T
        data = np.ravel(np.asarray(data)) * 1e-9
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
            in SI units.
        """
        factor = choclo.constants.VACUUM_MAGNETIC_PERMEABILITY / (4 * np.pi)
        xc, yc, zc = (
            i * MICROMETER_TO_METER for i in vdb.n_1d_arrays(self.location, 3)
        )
        x, y, z = (i * MICROMETER_TO_METER for i in vdb.n_1d_arrays(coordinates, 3))
        n_data = x.size
        n_params = 3
        jacobian = np.empty((n_data, n_params))
        for i in range(n_data):
            r = choclo.utils.distance_cartesian(x[i], y[i], z[i], xc, yc, zc)
            kernel_eu = choclo.point.kernel_eu(x[i], y[i], z[i], xc, yc, zc, r)
            kernel_nu = choclo.point.kernel_nu(x[i], y[i], z[i], xc, yc, zc, r)
            kernel_uu = choclo.point.kernel_uu(x[i], y[i], z[i], xc, yc, zc, r)
            jacobian[i, 0] = factor * kernel_eu
            jacobian[i, 1] = factor * kernel_nu
            jacobian[i, 2] = factor * kernel_uu
        return jacobian
