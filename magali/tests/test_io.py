# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the IO functions
"""

import numpy as np
import pooch
import scipy as sp

from .._input_output import random_unitary_vector, read_qdm_harvard


def test_read_qdm_harvard():
    "Try loading a sample dataset"
    path = pooch.retrieve(
        "doi:10.6084/m9.figshare.22965200.v1/Bz_uc0.mat",
        known_hash="md5:268bd3af5e350188d239ff8bd0a88227",
    )
    data = read_qdm_harvard(path)
    assert data.bz.size == 576_000


def test_random_unitary_vector():
    "Perform a shapiro test, in order to check normal distribution"
    r_vector = random_unitary_vector(np.deg2rad(50))

    assert (sp.stats.shapiro(r_vector[0]).pvalue >= 0.05) and (
        sp.stats.shapiro(r_vector[1]).pvalue >= 0.05
    )
