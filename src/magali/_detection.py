# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import harmonica as hm
import numpy as np
import skimage.exposure

from ._utils import _estimate_grid_spacing, total_gradient_amplitude_grid


def _calculate_blob_sizes(detected_scales, grid_spacing, size_multiplier):
    """
    Calculate the sizes of the blobs in μm.

    Parameters
    ----------
    detected_scales : array-like
        Detected blob scales (sigma values).
    grid_spacing : float
        Grid spacing between data points.
    size_multiplier : int
        Scaling factor for the detected blob sizes.

    Returns
    -------
    array-like
        Calculated sizes of the blobs in data units.
    """
    return detected_scales * np.sqrt(2) * grid_spacing * size_multiplier


def _generate_bounding_boxes(blob_sizes, blob_x_coords, blob_y_coords):
    """
    Generate bounding boxes around detected blobs.

    Parameters
    ----------
    blob_sizes : array-like
        Sizes of the detected blobs.
    blob_x_coords : array-like
        X coordinates of the detected blobs.
    blob_y_coords : array-like
        Y coordinates of the detected blobs.

    Returns
    -------
    list of lists
        Bounding boxes in data coordinates for each detected blob.
    """
    return [
        [x - size, x + size, y - size, y + size]
        for size, x, y in zip(blob_sizes, blob_x_coords, blob_y_coords)
    ]


def detect_anomalies(
    data,
    size_range,
    size_multiplier=2,
    num_scales=10,
    detection_threshold=0.5,
    overlap_ratio=0.5,
    border_exclusion=0,
    upward_continued=False,
    height_displacement=5,
    tga_calculated=False,
    stretched=False,
):
    """
    Detect anomalies using blob detection.

    Parameters
    ----------
    data : xr.DataArray
        Input data array with coordinates "x" and "y".
    size_range : tuple
        Minimum and maximum size of detected anomalies in µm.
    size_multiplier : int, optional
        Scaling factor for the detected blob sizes (default is 2).
    num_scales : int, optional
        Number of sigma values for the blob detection (default is 10). A sigma
        value represents the scale or size of the blobs that the algorithm
        will detect. Smaller sigma values correspond to smaller blobs, while
        larger sigma values correspond to larger blobs.
    detection_threshold : float, optional
        Detection threshold for the blob detection (default is 0.5). This
        parameter determines the sensitivity of the detection. A higher value
        means fewer blobs will be detected, and a lower value means more blobs
        will be detected.
    overlap_ratio : float, optional
        Overlap fraction for merging blobs (default is 0.5).
    border_exclusion : int, optional
        Border exclusion size in data units (default is 0). This parameter
        excludes blobs close to the border of the data array.
    upward_continued : boolean
        Indicate whether the data has already been upward continued or not.
    height_displacement : float
        The height displacement of upward continuation. For upward continuation,
        the height displacement should be positive. Its units are the same units
        of the grid coordinates.
    tga_calculated : boolean
        Indicate whether the total gradient amplitude has already been applied
        to the data.
    stretched : boolean
        Indicate whether the contrast stretching algorithm has already been
        applied to the data.

    Returns
    -------
    bounding_boxes : list of lists
        Bounding boxes of detected anomalies in data coordinates. Each
        bounding box corresponds to a detected blob, defined by the
        coordinates and size of the blob.
    """
    if not upward_continued:
        data_up = (
            hm.upward_continuation(data, height_displacement)
            .assign_attrs(data.attrs)
            .assign_coords(x=data.x, y=data.y)
            .assign_coords(z=data.z + height_displacement)
        )
    if not tga_calculated:
        tga = total_gradient_amplitude_grid(data_up)
    if not stretched:
        stretched_data = skimage.exposure.rescale_intensity(
            tga, in_range=tuple(np.percentile(tga, (1, 99)))
        )
    grid_spacing = _estimate_grid_spacing(stretched_data)
    min_sigma, max_sigma = [0.5 * size for size in size_range]

    y_indices, x_indices, detected_scales = skimage.feature.blob_log(
        stretched_data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=detection_threshold,
        num_sigma=num_scales,
        overlap=overlap_ratio,
        exclude_border=border_exclusion,
    ).T  # Transpose the output to separate y, x, and scale values

    blob_x_coords = data.x.values[x_indices.astype(int)]
    blob_y_coords = data.y.values[y_indices.astype(int)]

    blob_sizes = _calculate_blob_sizes(detected_scales, grid_spacing, size_multiplier)
    bounding_boxes = _generate_bounding_boxes(blob_sizes, blob_x_coords, blob_y_coords)
    return bounding_boxes
