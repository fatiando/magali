# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modeling code to calculate magnetic fields of finite sources.
"""

import harmonica as hm
import verde as vd
import xarray as xr

from ._units import coordinates_micrometer_to_meter


def dipole_magnetic(coordinates, dipole_coordinates, dipole_moments, field):
    """
    Compute the magnetic field produced by one or more dipoles.

    Uses :func:`harmonica.dipole_magnetic` for the calculations and handles the
    necessary unit versions from micrometers to meters.

    Parameters
    ----------
    coordinates : tuple of float
        Observation point coordinates in micrometers (μm).
    dipole_coordinates : tuple of float
        Dipole location coordinates in micrometers (μm).
    dipole_moments : tuple of float
        Dipole moment components (Am²).
    field : str
        Magnetic field that will be computed. The available fields are:

        - The full magnetic vector: ``b``
        - x component of the magnetic vector: ``b_x``
        - y component of the magnetic vector: ``b_y``
        - z (upward) component of the magnetic vector: ``b_z``

    Returns
    -------
    magnetic_field : array or tuple of arrays
        Computed magnetic field on every observation point in :math:`nT`.
        If ``field`` is set to a single component, then a single array with the
        computed magnetic field component will be returned.
        If ``field`` is set to ``"b"``, then a tuple containing the three
        components of the magnetic vector will be returned in the following
        order: ``b_x``, ``b_y``, ``b_z``.
    """
    coordinates_m = coordinates_micrometer_to_meter(coordinates)
    dipole_coordinates_m = coordinates_micrometer_to_meter(dipole_coordinates)
    translator = {"b_x": "b_e", "b_y": "b_n", "b_z": "b_u", "b": "b"}
    if field not in translator:
        message = (
            f"Invalid field argument '{field}'. Must be one of {translator.keys()}."
        )
        raise ValueError(message)
    return hm.dipole_magnetic(
        coordinates_m, dipole_coordinates_m, dipole_moments, field=translator[field]
    )


def dipole_magnetic_grid(
    region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moments, field
):
    """
    Generate a grid of the magnetic field of one or more dipoles.

    Uses :func:`magali.dipole_magnetic` to generate a regular grid of the
    magnetic field produced by a dipole model. The grid is returned as an
    :mod:`xarray` container type.

    Parameters
    ----------
    region : tuple = (x_min, x_max, y_min, y_max)
        The bounding box of the grid in micrometers (μm).
    spacing : float
        Grid spacing in micrometers (μm).
    sensor_sample_distance : float
        Distance of the sample from the grid in micrometers (μm).
    dipole_coordinates : tuple = (x, y, z)
        Arrays with the x, y, and z coordinates of the dipoles in micrometers
        (μm). The arrays can have any shape as long as they all have the same
        shape.
    dipole_moments : tuple = (mx, my, mz)
        Arrays with the x, y, and z components of the dipole moment of each
        dipole in Am². The arrays should have a shape compatible with the
        dipole coordinate arrays.
    field : str
        Magnetic field that will be computed. The available fields are:

        - The full magnetic vector: ``b``
        - x component of the magnetic vector: ``b_x``
        - y component of the magnetic vector: ``b_y``
        - z (upward) component of the magnetic vector: ``b_z``

    Returns
    -------
    grid : :class:`xarray.DataArray` or :class:`xarray.Dataset`
        Gridded magnetic field of the dipole model given. If *field* is
        a single component, the grid is returned as
        a :class:`xarray.DataArray`. If ``field ==  "b"``, the three components
        of the magnetic field will be returned as a :class:`xarray.Dataset`.
    """
    coordinates = vd.grid_coordinates(
        region=region,  # µm
        spacing=spacing,  # µm
        extra_coords=sensor_sample_distance,
    )
    data = dipole_magnetic(coordinates, dipole_coordinates, dipole_moments, field)
    dims = ("y", "x")
    coords = {
        "x": coordinates[0][0, :],
        "y": coordinates[1][:, 0],
        "z": (dims, coordinates[2]),
    }

    grid.x.attrs = {"units": "µm"}
    grid.y.attrs = {"units": "µm"}
    grid.bz.attrs = {"long_name": "vertical magnetic field", "units": "nT"}
    return data.bz
