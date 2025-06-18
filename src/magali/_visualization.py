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
    Plot bounding boxes on a matplotlib Axes.

    Used to visualize the bounding boxes of detected anomalies,
    such as those produced by :func:`magali._detection.detect_anomalies`.

    Parameters
    ----------
    bounding_boxes : list of list of float
        List of bounding boxes in data coordinates. Each bounding box is a list
        or array in the format [x_min, x_max, y_min, y_max].
    ax : matplotlib.axes.Axes, optional
        The matplotlib Axes on which to plot the bounding boxes. If None, uses
        the current Axes.
    edgecolor : str or tuple, default="k"
        Color of the bounding box edges.
    linewidth : float, default=2
        Width of the bounding box edges.
    **kwargs : dict, optional
        Additional keyword arguments passed to
        :class:`matplotlib.patches.Rectangle`, such as `linestyle` or `alpha`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object with the plotted bounding boxes.
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
