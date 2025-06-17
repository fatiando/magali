# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the visualization functions
"""

import matplotlib.pyplot as plt
import pytest

from magali._visualization import plot_bounding_boxes


@pytest.mark.mpl_image_compare
def plot_bounding_boxes_single_box():
    bounding_boxes = [[0, 1, 0, 1]]
    fig, ax = plt.subplots()
    plot_bounding_boxes(bounding_boxes, ax=ax)
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_bounding_boxes_multiple_boxes():
    bounding_boxes = [
        [0, 1, 0, 1],
        [2, 3, 2, 3],
        [-1, 0, -1, 0],
    ]
    fig, ax = plt.subplots()
    plot_bounding_boxes(bounding_boxes, ax=ax, edgecolor="red", linewidth=1.5)
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 4)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_bounding_boxes_with_kwargs():
    bounding_boxes = [[0, 2, 0, 2]]
    fig, ax = plt.subplots()
    plot_bounding_boxes(
        bounding_boxes, ax=ax, edgecolor="blue", linewidth=3, linestyle="--"
    )
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_bounding_boxes_without_ax():
    bounding_boxes = [[0, 1, 0, 1]]
    # Test the behavior when ax is None
    plot_bounding_boxes(bounding_boxes)
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    fig = plt.gcf()
    return fig
