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
    meter_to_micrometer,
    coordinates_micrometer_to_meter,
    nanotesla_to_tesla,
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
        r5 = r2**2.5

        dBz_dx = factor * 3 * (
            (mx * (5 * dx**2 * dz - r2 * dz) +
            my * (5 * dx * dy * dz) +
            mz * (5 * dx * dz**2 - r2 * dx))
        ) / r5

        dBz_dy = factor * 3 * (
            (mx * (5 * dx * dy * dz) +
            my * (5 * dy**2 * dz - r2 * dz) +
            mz * (5 * dy * dz**2 - r2 * dy))
        ) / r5

        dBz_dz = factor * 3 * (
            (mx * (5 * dx * dz**2 - r2 * dx) +
            my * (5 * dy * dz**2 - r2 * dy) +
            mz * (5 * dz**3 - 3 * r2 * dz))
        ) / r5

        dBz_dmx = factor * (3 * dx * dz) / r2**2.5
        dBz_dmy = factor * (3 * dy * dz) / r2**2.5
        dBz_dmz = factor * (3 * dz**2 - r2) / r2**2.5

        result[i, 0] = dBz_dx
        result[i, 1] = dBz_dy
        result[i, 2] = dBz_dz
        result[i, 3] = dBz_dmx
        result[i, 4] = dBz_dmy
        result[i, 5] = dBz_dmz



class MagneticDipoleBz:
    def __init__(self, initial):
        self.initial = np.array(initial, dtype=float)
        self.location_ = None
        self.dipole_moment_ = None
        self.parameters_ = None
        self.jacobian = None
    
    def forward(self, coordinates, parameters):
        """
        Compute the predicted Bz for a given dipole model.

        Parameters
        ----------
        coordinates : tuple = (x, y, z)
            Observation coordinates in µm.
        parameters : tuple
            Dipole parameters (x, y, z, mx, my, mz).

        Returns
        -------
        Bz : array
            Predicted Bz in nT at the given coordinates.
        """
        location = parameters[:3]
        moment = parameters[3:]
        return dipole_bz(coordinates, location, moment)
    
    
    def _jacobian(self, coordinates, parameters):
        xc, yc, zc = parameters[:3]
        mx, my, mz = parameters[3:]
        x,y,z = coordinates
        n_data = x.size
        n_params = 6
        jacobian = np.empty((n_data, n_params))
        jacobian_kernel_jit(x, y, z, xc, yc, zc, mx, my, mz, jacobian)
        return jacobian

    def fit(self, coordinates, data, maxiter=100, damping=1e-3, tol=1e-10):
        coordinates, data = check_fit_input(coordinates, data)
        data = data.ravel()
        coordinates = coordinates_micrometer_to_meter(coordinates) 
        x0, y0, z0, mx0, my0, mz0 = self.initial
        params = np.array(
            [
                *coordinates_micrometer_to_meter((x0, y0, z0)),
                mx0,
                my0,
                mz0,
            ],
            dtype=float,
        )

        for _ in range(maxiter):
            predicted = self.forward(coordinates, params)
            residuals = data - predicted
            jacobian = self._jacobian(coordinates, params)

            hessian = jacobian.T @ jacobian
            gradient = jacobian.T @ residuals
            step = np.linalg.solve(hessian + damping * np.identity(6), gradient)

            params += step

            if np.linalg.norm(step) / np.linalg.norm(params) < tol:
                break

        self.location_ = params[:3]
        self.dipole_moment_ = params[3:]
        return self




# Compile the Jacobian calculation. Doesn't use this as a decorator so that we
# can test the pure Python function and get coverage information about it.
jacobian_jit = numba.jit(_jacobian, nopython=True, parallel=True)

jacobian_kernel_jit = numba.jit(_jacobian_kernel, nopython=True, parallel=True)
