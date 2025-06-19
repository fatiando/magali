# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#


import numpy as np


def _calculate_angular_distance(
    inclination_one, declination_one, inclination_two, declination_two
):
    r"""
    Calculate the Angular distance between two vectors.

    Parameters
    ----------
    inclination_one: float
        Inclination of the first vector in degrees.
    declination_one: float
        Declination of the first vector in degrees.
    inclination_two: float
        Inclination of the second vector in degrees.
    declination_two: float
        Declination of the second vector in degrees.

    Returns
    -------
    angular_distance : np.float64
        Angular distance between two vectors in degrees.
    """
    angular_distance = np.arccos(
        np.sin(np.radians(inclination_two)) * np.sin(np.radians(inclination_one))
        + np.cos(np.radians(inclination_two))
        * np.cos(np.radians(inclination_one))
        * np.cos(np.radians(abs(declination_two - declination_one)))
    )

    return np.degrees(angular_distance)


def variance(inclination, declination, inclination_mean, declination_mean):
    r"""
    Calculate angular variance.

    Parameters
    ----------
    inclination: array
        Inclination value of vectors in degrees.
    declination: array
        Declination value of vectors in degrees.
    inclination_mean: float
        Mean inclination in degrees.
    declination_mean: float
        Mean declination in degrees.

    Returns
    -------
    angular_distance : np.float64
        Angular distance between two vectors in degrees.
    """
    distance = _calculate_angular_distance(
        inclination, declination, inclination_mean, declination_mean
    )
    sum_delta_angle = np.sum(distance**2)

    variance = (1 / (len(inclination) - 1)) * sum_delta_angle

    return variance
