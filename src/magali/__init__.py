# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Import functions/classes to make the public API.
"""

from ._detection import detect_anomalies
from ._input_output import read_qdm_harvard
from ._inversion import MagneticMomentBz, NonlinearMagneticDipoleBz
from ._synthetic import dipole_bz, dipole_bz_grid, random_directions
from ._units import (
    coordinates_micrometer_to_meter,
    meter_to_micrometer,
    nanotesla_to_tesla,
    tesla_to_nanotesla,
)
from ._utils import (
    gradient,
    total_gradient_amplitude_grid,
)
from ._validation import check_coordinates, check_fit_input
from ._version import __version__
from ._visualization import plot_bounding_boxes

# Append a leading "v" to the generated version by setuptools_scm
__version__ = f"v{__version__}"
