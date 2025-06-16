# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions for plotting and visualization of data.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def plot_bounding_boxes(bounding_boxes, ax=None, edgecolor="k", linewidth=2, **kwargs):
    """
    Plot the bounding boxes detected from :func:`magali._detection.detect_anomalies`.

    Parameters
    ----------
    bounding_boxes : list of lists
        Bounding boxes of detected anomalies in data coordinates. Each
        bounding box corresponds to a detected blob, defined by the
        coordinates and size of the blob.
    title : str
        Title of the plot.
    **kwargs :
        Additional keyword arguments passed to matplotlib.patches.Rectangle.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object of the plot.
    """
    if ax is None:
        ax = plt.gca()
    for box in bounding_boxes:
        rect = mpatches.Rectangle(
            xy=[box[0], box[2]],
            width=box[1] - box[0],
            height=box[3] - box[2],
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            **kwargs,
        )
        ax.add_patch(rect)
    return ax
