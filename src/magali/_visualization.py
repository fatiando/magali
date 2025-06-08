# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions for plotting and visualization of data
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def plot_detected_sources(
    grid, bounding_boxes, cmap="seismic", title="Detected Magnetic Sources"
):
    """
    Plot a 2D grid and overlay rectangles for detected boxdows.

    Parameters
    ----------
    grid : xr.DataArray
        The 2D grid to be plotted.
    bounding_boxes : list of lists
        Bounding boxes of detected anomalies in data coordinates. Each
        bounding box corresponds to a detected blob, defined by the
        coordinates and size of the blob.
    cmap : str
        Colormap for the background grid.
    title : str
        Title of the plot.
    """
    fig, ax = plt.subplots()
    grid.plot.pcolormesh(ax=ax, cmap=cmap)
    for box in bounding_boxes:
        rect = mpatches.Rectangle(
            xy=[box[0], box[2]],
            width=box[1] - box[0],
            height=box[3] - box[2],
            edgecolor="k",
            fill=False,
            linewidth=2,
        )
        ax.add_patch(rect)
    ax.set_title(title)
    plt.show()
