# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions to check and conform inputs into our functions and classes.
"""

import numpy as np


def check_fit_input(coordinates, data):
    """
    Validate the inputs to the fit method of inversion classes.

    Checks that the coordinates and data all have the same shape. Ravel all
    inputs to make sure they are 1D arrays. Transform lists to numpy arrays if
    necessary. Coordinates must contain 3 arrays for the x, y, and
    z coordinates.

    Parameters
    ----------
    coordinates : tuple = (x, y, z)
        Tuple of arrays with the x, y, z coordinates of each point. Arrays can
        be Python lists or any numpy-compatible array type. Arrays can be of
        any shape but must all have the same shape.
    data : array
        The data values of each data point. Can be a Python list or any
        numpy-compatible array type. Can be of any shape but must have the same
        shape as the coordinates.

    Returns
    -------
    coordinates, data : tuple
        The validated inputs in the same order. All will be raveled to 1D
        arrays and converted to numpy arrays if needed.

    Raises
    ------
    ValueError
        If the arrays don't all have the same shape or if there are fewer than
        3 coordinates.
    """
    coordinates = check_coordinates(coordinates)
    data = np.atleast_1d(data)
    if data.shape != coordinates[0].shape:
        message = (
            "Data array must have the same shape as coordinate arrays. "
            f"Given data with shape {data.shape} and coordinates with shape "
            f"{coordinates[0].shape}."
        )
        raise ValueError(message)
    return tuple(c.ravel() for c in coordinates), data.ravel()


def check_coordinates(coordinates):
    """
    Check that a given tuple of coordinate arrays is valid.

    Make sure that there are 3 arrays and that all arrays have the same shape.
    Convert any lists/tuples to arrays.

    Parameters
    ----------
    coordinates : tuple = (x, y, z)
        Tuple of arrays with the x, y, z coordinates of each point. Arrays can
        be Python lists or any numpy-compatible array type. Arrays can be of
        any shape but must all have the same shape.

    Returns
    -------
    coordinates : tuple = (x, y, z)
        The validated coordinate numpy arrays.

    Raises
    ------
    ValueError
        If the arrays don't all have the same shape or if there are fewer than
        3 coordinates.
    """
    if len(coordinates) != 3:
        message = (
            "Three coordinates are required (x, y, and z), but only "
            f"{len(coordinates)} were given."
        )
        raise ValueError(message)
    coordinates = tuple(np.atleast_1d(i) for i in coordinates)
    shapes = [i.shape for i in coordinates]
    if not all(shape == shapes[0] for shape in shapes):
        message = (
            f"Invalid coordinate arrays. "
            "All arrays must have the same shape. "
            f"Given shapes: {shapes}"
        )
        raise ValueError(message)
    return coordinates
