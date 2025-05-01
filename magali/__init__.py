# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
from ._detection import detect_anomalies
from ._input_output import read_qdm_harvard
from ._synthetic import dipole_bz, dipole_bz_grid, random_directions
from ._utils import (
    gradients,
    total_gradient_amplitude,
    total_gradient_amplitude_grid,
)
from ._version import __version__
from ._inversion import MagneticMomentBz
from ._constants import MICROMETER_TO_METER, METER_TO_MICROMETER, TESLA_TO_NANOTESLA