# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the IO functions
"""

import pooch

from .._io import read_qdm_harvard


def test_read_qdm_harvard():
    "Try loading a sample dataset"
    path = pooch.retrieve(
        "doi:10.6084/m9.figshare.22965200.v1/Bz_uc0.mat",
        known_hash="md5:268bd3af5e350188d239ff8bd0a88227",
    )
    data = read_qdm_harvard(path)
    assert data.bz.size == 576_000
