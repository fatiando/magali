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
import xarray as xr

from .._input_output import read_qdm_harvard


def test_read_qdm_harvard():
    "Try loading a sample dataset"
    fname = ensaio.fetch_morroco_speleothem_qdm(version=1, file_format="matlab")
    bz = read_qdm_harvard(fname)

    # Test units
    assert bz.x.units == "µm"
    assert bz.y.units == "µm"
    assert bz.z.units == "µm"
    assert bz.units == "nT"

    # Test array sizes
    assert bz.size == 576_000
    assert bz.x.size == 960
    assert bz.y.size == 600
    assert bz.z.size == 576000
    assert bz.size == 576000

    # Test data name
    assert bz.long_name == "vertical magnetic field"
    assert bz.z.long_name == "sensor sample distance"

    # Test if bz is a DataArray
    assert isinstance(bz, xr.DataArray)
    assert isinstance(bz.x, xr.DataArray)
    assert isinstance(bz.y, xr.DataArray)
    assert isinstance(bz.z, xr.DataArray)
