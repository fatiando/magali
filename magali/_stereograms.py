# Copyright (c) 2024 The Magali Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#


import matplotlib.pyplot as plt
import numpy as np

from ._utils import angles_to_vector, vector_to_angles


def equal_area_projection(vectors):
    """
    Perform an equal-area projection of 3D vectors onto a 2D plane.

    Parameters
    ----------
    vectors : ndarray of shape (n, 3)
        Array of 3D vectors to be projected. Each row represents a vector
        with components [x, y, z].

    Returns
    -------
    xy_projected : ndarray of shape (n, 3)
        Array containing the 2D projected coordinates each vector.
        The first two columns represent the x and y coordinates on the
        2D plane. The third column contains the amplitude, where the
        sign indicates whether the inclination is positive (downward)
        or negative (upward).
    """

    norm = np.linalg.norm(vectors, axis=1)
    vectors_unitary = vectors / norm[:, np.newaxis]
    inclinations, declinations, amplitudes = vector_to_angles(vectors)

    xy_projected = np.zeros((len(vectors), 3))
    for i, _ in enumerate(vectors_unitary):
        r = np.sqrt(1 - np.abs(vectors_unitary[i, 2])) / np.sqrt(
            vectors_unitary[i, 0] ** 2 + vectors_unitary[i, 1] ** 2
        )
        xy_projected[i, 0] = r * vectors_unitary[i, 1]
        xy_projected[i, 1] = r * vectors_unitary[i, 0]
        xy_projected[i, 2] = amplitudes[i] if inclinations[i] >= 0 else -amplitudes[i]

    return xy_projected


class StereographicProjection:
    def __init__(self, vectors):
        self.vectors = vectors

    def plot(
        self,
        ax=None,
        cmap="inferno",
        cmap_norm=plt.Normalize,
        vmin=None,
        vmax=None,
        label="",
        add_ticks=True,
        draw_cross=True,
        add_radial_grid=True,
        facecolor="#ffffff00",
        add_legend=False,
        **kwargs,
    ):
        r"""
        Plot a stereographic projection with a radius of 1.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis on which to plot. If None, a new figure and axis will
            be created.
        cmap : str, optional
            Colormap to be used for the plotted data. Default is 'inferno'.
        cmap_norm : matplotlib.colors.Normalize, optional
            Normalization function for the colormap. Default is plt.Normalize.
        vmin : float, optional
            Minimum data value for colormap normalization. If None,
            the minimum value of the data is used.
        vmax : float, optional
            Maximum data value for colormap normalization. If None,
            the maximum value of the data is used.
        label : str, optional
            Label for the plotted data. Default is an empty string.
        draw_cross : bool, optional
            Whether to draw the central cross on the plot. Default is True.
        add_radial_grid : bool, optional
            Whether to add a radial grid to the plot. Default is True.
        facecolor : str, optional
            The facecolor of the plot background. Default is
            transparent ("#ffffff00").
        add_legend : bool, optional
            The legend of the plot. Default is False.
        kwargs : dict, optional
            Additional keyword arguments passed to the scatter function.

        Returns
        -------
        mappable : matplotlib.cm.ScalarMappable
            The ScalarMappable object, which can be used to create a colorbar.
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        # Add a face color
        background_circle = plt.Circle((0, 0), 1, color=facecolor, zorder=-2)
        ax.add_artist(background_circle)

        # Draw the great circle
        circle = plt.Circle((0, 0), 1, color="black", fill=False, zorder=3)
        ax.add_artist(circle)

        # Define the clipping area
        clip_path = plt.Circle((0, 0), 1, transform=ax.transData)

        if add_ticks:
            # Add ticks
            ax.text(-0.025, 1.05, "0°")
            ax.text(-0.05, -1.075, "180°")
            ax.text(-1.15, -0.025, "270°")
            ax.text(1.025, -0.025, "90°")

        if label and not label.endswith(" "):
            label = label + " "
            # Draw the central cross if requested
            if draw_cross:
                hline = ax.axhline(y=0, color="black", zorder=-1)
                hline.set_clip_path(clip_path)
                vline = ax.axvline(x=0, color="black", zorder=-1)
                vline.set_clip_path(clip_path)

        # Draw the radial grid
        if add_radial_grid:
            for rad_dec in range(0, 360, 10):  # Grid lines every 10 degrees
                rad_inc = np.linspace(0, 90, 1000)  # Inclinations from 0 to 90 degrees
                # Generate the radial unitary vectors
                radial_vector = angles_to_vector(rad_inc, rad_dec, 1)
                # Project the radial vectors
                radial_projected = equal_area_projection(radial_vector)

                # Plot each radial grid line
                ax.plot(
                    radial_projected[:, 1],
                    radial_projected[:, 0],
                    color="gray",
                    zorder=-2,
                    lw=0.5,
                )
            for circ_inc in range(0, 90, 10):  # Isoinclination lines every 10 degrees
                circ_dec = np.linspace(
                    0, 360, 1000
                )  # Declinations from 0 to 360 degrees
                # Generate the circular unitary vectors
                circle_vector = angles_to_vector(circ_inc, circ_dec, 1)
                # Project the radial vectors
                circle_projected = equal_area_projection(circle_vector)

                # Plot each radial grid line
                ax.plot(
                    circle_projected[:, 1],
                    circle_projected[:, 0],
                    color="gray",
                    zorder=-2,
                    lw=0.5,
                )

        # Calculate the equal area projection
        xy_projected = equal_area_projection(self.vectors)

        # Generate colors based on the amplitude values
        norm = cmap_norm(vmin=vmin, vmax=vmax)
        colors = plt.colormaps[cmap](norm(abs(xy_projected[:, 2])))

        # Plotting the data
        positive_inc = xy_projected[:, 2] > 0
        scatter_pos = ax.scatter(
            xy_projected[:, 1][positive_inc],
            xy_projected[:, 0][positive_inc],
            c=colors[positive_inc],
            edgecolors="#333333",
            label=f"{label}$I > 0$",
            **kwargs,
        )
        scatter_pos.set_clip_path(clip_path)

        scatter_neg = ax.scatter(
            xy_projected[:, 1][~positive_inc],
            xy_projected[:, 0][~positive_inc],
            c="#ffffff00",
            edgecolors=colors[~positive_inc],
            label=rf"{label}$I \leq 0$",
            **kwargs,
        )
        scatter_neg.set_clip_path(clip_path)

        # Configure the axis
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if add_legend:
            ax.legend()
        for spine in ax.spines.values():
            spine.set_visible(False)

        # To generate colorbar if necessary
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        return mappable
