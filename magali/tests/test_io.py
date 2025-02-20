# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the IO functions
"""

import ensaio

from .._input_output import read_qdm_harvard


def test_read_qdm_harvard():
    "Try loading a sample dataset"
    fname = ensaio.fetch_morroco_speleothem_qdm(version=1, file_format="matlab")
    bz = read_qdm_harvard(fname)
    assert bz.size == 576_000
