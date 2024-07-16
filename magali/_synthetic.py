# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions to generate synthetic data
"""

import harmonica as hm
import numpy as np
import verde as vd


def random_directions(
    inclination, declination, dispersion_angle, size, random_state=None
):
    r"""
    Creates random directions around a preferred direction.

    Parameters
    ----------
    inclination : float
        Inclination of the preferred direction in degrees.
    declination : float
        Declination of the preferred direction in degrees.
    dispersion_angle : float
        Dispersion angle that defines a region on the surface of a sphere.
        Also corresponds to the variance of the generated directions.
    size : int
        Number of random directions to be generated.
    random_state : numpy.random.RandomState or an int seed
        A random number generator used to define the state of the random
        permutations. Use a fixed seed to make sure computations are
        reproducible. Use ``None`` to choose a seed automatically (resulting in
        different numbers with each run).

    Returns
    -------
    inclination : numpy.array
        Inclination of the generated directions in degrees.
    declination : numpy.array
        Declination of the generated directions in degrees.

    Notes
    -----
    We calculate the azimuth (\alpha) via a random uniform distribution
    ranging from 0° to 360°, to represent the equal probability for each
    value in a stereographic polar projection.
    The distance to the pole (\psi) for each vector is obtained with a normal
    distribution.
    Given the values of azimuth (\alpha) and distance to the pole (\psi), we
    calculate the catesian coordinates

    .. math::
        x = \sin(\psi) * cos(\alpha)
    .. math::
        y = np.sin(\psi) * sin(\alpha)
    .. math::
        z = np.cos(\psi)

    The rotation of the pole is performed to the preferred direction using the
    specified values of inclination and declination to obtain the rotation
    vector. With :math:`\theta=90°+inclination` and
    :math:`\phi=90°-declination` we calculate the rotation
    vector by

    .. math::
        \mathbf{u}=\begin{bmatrix}
        x\\y\\z
        \end{bmatrix}\\
    .. math::
        \mathbf{R}_z(\phi)\mathbf{R}_y(\theta)\mathbf{u}=\begin{bmatrix}
        \cos(\theta)&0&\sin(\theta)\\
        0&1&0\\
        -\sin(\theta)&0&\cos(\cos)
        \end{bmatrix}\begin{bmatrix}
        \cos(\phi)&-\sin(\phi)&0\\
        \sin(\phi)&\cos(\phi)&0\\
        0&0&1
        \end{bmatrix}\begin{bmatrix}
        x\\y\\z
        \end{bmatrix}\\
    .. math::

        \mathbf{R}_z(\phi) \mathbf{R}_y(\theta)\mathbf{u}  =   \begin{bmatrix}
        \cos(\phi)(X\cos(\theta)+Z\sin(\theta))-Y\sin(\phi) \\
        \sin(\phi)(X\cos(\theta)+Z\sin(\theta))+Y\cos(\phi) \\
        -X\sin(\theta)+Z\cos(\theta)
        \end{bmatrix}

    in which :math:`\mathbf{R}_z(\phi)` and :math:`\mathbf{R}_z(\phi)` are the
    rotation matrices on the z and y axes, respctively.
    """
    # Set random number generator
    rng = np.random.default_rng(random_state)

    azimuth = np.deg2rad(rng.uniform(0, 360, size))
    distance = rng.normal(0, np.deg2rad(dispersion_angle), size)
    x = np.sin(distance) * np.cos(azimuth)
    y = np.sin(distance) * np.sin(azimuth)
    z = np.cos(distance)

    x_r, y_r, z_r = _rotate_vector(x, y, z, inclination, declination)

    _, directions_inclination, directions_declination = hm.magnetic_vec_to_angles(
        x_r, y_r, z_r
    )
    return directions_inclination, directions_declination


def _rotate_vector(x, y, z, inclination, declination):
    """
    Rotates vectors using its cartesian coordines towards specific direction.

    Parameters
    ----------
    inclination : float
        Inclination of the preferred direction in degrees.
    declination : float
        Declination of the preferred direction in degrees.
    x : numpy.array
        x coordinates of vectors to be rotated.
    y : numpy.array
        y coordinates of vectors to be rotated.
    z : numpy.array
        z coordinates of vectors to be rotated.

    Returns
    -------
    x_r : numpy.array
        x coordinates of rotated vectors.
    y_r : numpy.array
        y coordinates of rotated vectors.
    z_r : numpy.array
        z coordinates of rotated vectors.
    """
    theta = np.deg2rad(90 + inclination)
    phi = np.deg2rad(90 - declination)

    # R_z(phi)*R_y(theta)
    x_r = np.cos(phi) * (x * np.cos(theta) + z * np.sin(theta)) - y * np.sin(phi)
    y_r = np.sin(phi) * (x * np.cos(theta) + z * np.sin(theta)) + y * np.cos(phi)
    z_r = -x * np.sin(theta) + z * np.cos(theta)
    return x_r, y_r, z_r


def amplitude_lognormal_distribution(size, mean=0, sigma=1.35, scale=1e-14):
    r"""
    Generate amplitude values in a lognormal distribution.

    Parameters
    ----------
    mean : float or array_like of floats, optional
        Mean value of the underlying normal distribution. Default is 0.

    sigma : float or array_like of floats, optional
        Standard deviation of the underlying normal distribution. Must be
        non-negative. Default is 1.

    size : int
        Number of values to be generated.

    Returns
    -------
    amplitude : ndarray or scalar
        Samples drawn from the parameterized log-normal distribution,
        representing random amplitude values. The output array is scaled
        to represent values in nT (nanoteslas).
    """
    amplitude = np.random.lognormal(mean, sigma, size)
    amplitude = amplitude * scale
    return amplitude


def random_coordinates(sources_region, number_of_sources, seed=5):
    r"""
    Generate random coordinates within a specified region.

    This function generates a specified number of random (x, y, z) coordinates
    within given boundaries. The coordinates are generated using a
    reproducible random number generator.

    Parameters
    ----------
    sources_region : list of float
        A list containing the limits for the x, y, and z coordinates.
        Format: [x_min, x_max, y_min, y_max, z_min, z_max].
        
        - x_min, x_max: The minimum and maximum bounds for the x coordinates.
        - y_min, y_max: The minimum and maximum bounds for the y coordinates.
        - z_min, z_max: The minimum and maximum bounds for the z coordinates.

    number_of_sources : int
        The number of sources for which coordinates will be generated.

    seed : int, optional
        Seed value for the random number generator to ensure reproducibility
        of results. Default is 5.

    Returns
    -------
    coordinates : numpy.ndarray
        A 2D numpy array of shape (3, number_of_sources) containing the
        generated coordinates. The rows represent x, y and z coordinates
        respectively.
    """
    np.random.seed(seed)
    coordinates = (
        np.random.randint(sources_region[0], sources_region[1], number_of_sources),
        np.random.randint(sources_region[2], sources_region[3], number_of_sources),
        np.random.randint(sources_region[4], sources_region[5], number_of_sources),
    )
    return np.asarray(coordinates)


def generate_dipoles_grid(
    region,
    spacing,
    sensor_sample_distance,
    dipole_coordinates,
    amplitude,
    inclination,
    declination,
    field,
):
    """
    Creates a regular grid of observation points.

    Parameters
    ----------
    region: list
        The boundaries of a given region in Cartesian coordinates in µm.
    spacing : float, tuple = (s_north, s_east), or None
        The number of blocks in the South-North and West-East directions,
        respectively. If None, then spacing must be provided.
    sensor_sample_distance : float
        Distance between sensor and sample in µm.
    dipole_coordinates : numpy.ndarray
        A 2D numpy array of shape (3, number_of_sources) containing the
        generated coordinates.The first row represents x coordinates.The second
        row represents y coordinates. The third row represents z coordinates.
    amplitude : ndarray or scalar
        Samples drawn from the parameterized log-normal distribution,
        representing random amplitude values in nT (nanoteslas).
    inclination : numpy.array
        Inclination of the dipoles directions in degrees.
    declination : numpy.array
        Declination of the dipoles directions in degrees.
    field : str
        Magnetic field that will be computed. The available fields are:

        - The full magnetic vector: ``b``
        - Easting component of the magnetic vector: ``b_e``
        - Northing component of the magnetic vector: ``b_n``
        - Upward component of the magnetic vector: ``b_u``

    Returns
    -------
    grid : (xarray.DataArray)
        xarray.DataArray containing the grid, its coordinates and header
        information.



    """
    grid_coordinates = vd.grid_coordinates(
        region=region,
        spacing=spacing,
        extra_coords=sensor_sample_distance,
    )

    x, y, z = hm.magnetic_angles_to_vec(amplitude, inclination, declination)

    dipole_moments = (x, y, z)

    # Coordinates are multiplied by 1.0E-6 to fix scale
    bz = hm.dipole_magnetic(
        np.asarray(grid_coordinates) * 1.0e-6,
        np.asarray(dipole_coordinates) * 1.0e-6,
        dipole_moments,
        field,
    )

    grid = vd.make_xarray_grid(
        grid_coordinates, bz, data_names=["bz"], dims=("y", "x"), extra_coords_names="z"
    )
    grid.x.attrs = {"units": "µm"}
    grid.y.attrs = {"units": "µm"}
    grid.bz.attrs = {"long_name": "vertical magnetic field", "units": "nT"}

    return grid
