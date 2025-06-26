# Copyright (c) 2023 Gelson Ferreira de Souza-Junior, Leonardo Uieda.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
"""
Functions for performing the processing and inversion of the microscopy data.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.io
import skimage.feature
import skimage.exposure
import skimage.restoration
import numba
import harmonica as hm
import verde as vd
import choclo

from scipy.fftpack import fftshift, ifft2, ifft

TESLA_TO_NANOTESLA = 1e9
MICROMETER_TO_METER = 1e-6
METER_TO_MICROMETER = 1e6


def enhance_image(image, percentile=(5, 95)):
    """
    Apply constrast stretching and denoising to enhance an image.
    """
    enhanced = skimage.restoration.denoise_nl_means(
        skimage.exposure.rescale_intensity(
            image, in_range=tuple(np.percentile(image, percentile)),
        ),
    )
    return enhanced


def gaussian_noise(error, shape, seed=None):
    """
    Generate a pseudo-random noise array of the given shape.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=error, size=shape)
    return noise


def nondipolar_source(
    xc, yc, zc, position_std, inclination, declination, angle_std, 
    amplitude, amplitude_std, n_particles=200, seed=None,
):
    """
    Generate a non-dipolar source as an ensemble of random dipoles
    """
    rng = np.random.default_rng(seed)
    # Make sure z is always < 0 (dipole is inside the sample)
    dipole_coordinates = (
        rng.normal(xc, position_std, n_particles),
        rng.normal(yc, position_std, n_particles),
        -np.abs(rng.normal(zc, position_std, n_particles)),
    )
    dipole_moments = angles_to_vector(
        rng.normal(inclination, angle_std, n_particles),
        rng.normal(declination, angle_std, n_particles),
        rng.normal(amplitude, amplitude_std, n_particles),
    )
    return (dipole_coordinates, dipole_moments)


def angles_to_vector(inclination, declination, amplitude):
    """
    Generate a 3-component vector from inclination, declination, and amplitude

    Inclination is positive downwards and declination is the angle with the y
    component. The vector has x, y, and z (upward) Cartesian components.

    Parameters
    ----------
    inclination : float or array
        The inclination values in degrees.
    declination : float or array
        The declination values in degrees.
    amplitude : float or array
        The vector amplitude values.

    Returns
    -------
    vector : 1D or 2D array
        The calculated x, y, z vector components. 1D if it's a single vector.
        If N vectors are calculated, the "vector" will have shape (N, 3) with
        each vector in a row of the array.
    """
    inclination = np.radians(inclination)
    declination = np.radians(declination)
    amplitude = np.asarray(amplitude)
    sin_inc = np.sin(-inclination)
    cos_inc = np.cos(-inclination)
    sin_dec = np.sin(declination)
    cos_dec = np.cos(declination)
    x = cos_inc * sin_dec * amplitude
    y = cos_inc * cos_dec * amplitude
    z = sin_inc * amplitude
    return np.transpose([x, y, z])


def vector_to_angles(vector):
    """
    Generate inclination, declination, and amplitude from a 3-component vector

    Inclination is positive downwards and declination is the angle with the y
    component. The vector has x, y, and z (upward) Cartesian components.

    Parameters
    ----------
    vector : 1D or 2D array
        The x, y, z vector components. Can be a 1D array for a single vector
        or 2D for multiple. If 2D, then each vector should be a row of the
        array.

    Returns
    -------
    inclination : float or array
        The inclination values in degrees.
    declination : float or array
        The declination values in degrees.
    amplitude : float or array
        The vector amplitude values.
    """
    vector = np.asarray(vector)
    x, y, z = vector.T
    amplitude = np.sqrt(x**2 + y**2 + z**2)
    inclination = -np.degrees(np.arctan2(z, np.hypot(x, y)))
    declination = np.degrees(np.arctan2(x, y))
    return inclination, declination, amplitude


def dipole_bz(coordinates, dipole_coordinates, dipole_moments):
    """
    Calculate the z component (upward) of the magnetic field of dipoles

    Units:

    * Coordinates: µm
    * Dipole moment: A.m²
    * Magnetic field: nT

    Parameters
    ----------
    coordinates : tuple of arrays
        The coordinates of the computation/data points. Should be a tuple with
        arrays representing the x, y, and z coordinates of each point.
    dipole_coordinates : tuple of arrays
        The coordinates of the dipoles. Should be a tuple with arrays
        representing the x, y, and z coordinates of each dipole.
    dipole_moments : list or 2D-array
        A list of 3-component 1D arrays representing the dipole moment of each
        dipole. Can also be a 2D array with M rows (M = number of dipoles) and
        3 columns.

    Returns
    -------
    bz : array
        The calculated bz (positive upward) in nT. The array will have the
        same shape as the input coordinate arrays.
    """
    # Save the data coordinate shape so we can reshape the bz afterwards
    data_shape = coordinates[0].shape
    # Make sure coordinates are 1D arrays and the dipole moment is 2D also
    # convert all units to SI
    coordinates = [np.asarray(c).ravel() * MICROMETER_TO_METER for c in coordinates]
    dipole_coordinates = [
        np.asarray(c).ravel() * MICROMETER_TO_METER for c in dipole_coordinates
    ]
    dipole_moments = np.asarray(dipole_moments)
    if dipole_moments.ndim == 1:
        dipole_moments = np.array([dipole_moments])
    # Initialize and allocate the output array for bz
    bz = np.zeros(coordinates[0].shape)
    _dipole_bu_fast(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        dipole_coordinates[0],
        dipole_coordinates[1],
        dipole_coordinates[2],
        dipole_moments,
        bz,
    )
    bz *= TESLA_TO_NANOTESLA
    return bz.reshape(data_shape)


@numba.jit(nopython=True, parallel=True)
def _dipole_bu_fast(e, n, u, de, dn, du, dipole_moments, bu):
    """
    This is the bit that runs the fast for-loops
    """
    for i in numba.prange(e.size):
        for j in range(de.size):
            result = choclo.dipole.magnetic_u(
                easting_p=e[i],
                northing_p=n[i],
                upward_p=u[i],
                easting_q=de[j],
                northing_q=dn[j],
                upward_q=du[j],
                magnetic_moment_east=dipole_moments[j, 0],
                magnetic_moment_north=dipole_moments[j, 1],
                magnetic_moment_up=dipole_moments[j, 2]
            )
            bu[i] += result


def dipole_moment_inversion(data, dipole_coordinates):
    """
    Estimate the dipole moment through linear inversion.

    Parameters
    ----------
    data : xarray.DataArray
        The observed vertical magnetic field grid. Must have x, y, and z as
        coordinates.
    dipole_coordinates : 1D array
        The (x, y, z) coordinates of the dipole.

    Returns
    -------
    dipole_moment : 1D array
        The estimated dipole moment vector.
    covariance : 2D array
        The estimated covariance matrix of the dipole moment vector.
    r2 : float
        The R² coefficient of determination of the inversion.
    """
    # Make it a dataset so we can be sure the variable name is "bz"
    table = vd.grid_to_table(data.to_dataset(name="bz"))
    # Verde drops non-dimension coordinates so we have to add z back.
    # This is a bug in Verde.
    table["z"] = data.z.values.ravel()
    n_data = table.shape[0]
    n_params = 3
    A = np.empty((n_data, n_params))
    d = table.bz.values / TESLA_TO_NANOTESLA
    # Fill the Jacobian using a fast calculation with numba
    _dipole_jacobian_fast(
        table.x.values * MICROMETER_TO_METER,
        table.y.values * MICROMETER_TO_METER,
        table.z.values * MICROMETER_TO_METER,
        dipole_coordinates[0] * MICROMETER_TO_METER,
        dipole_coordinates[1] * MICROMETER_TO_METER,
        dipole_coordinates[2] * MICROMETER_TO_METER,
        A,
    )
    hessian = A.T @ A
    neg_gradient = A.T @ d
    dipole_moment = scipy.linalg.solve(hessian, neg_gradient, assume_a="pos")
    residuals = d - A @ dipole_moment
    residuals_sum_sq = np.sum(residuals**2)
    # Estimate of the true error variance (since we'll never know it)
    chi_squared = residuals_sum_sq / (n_data - n_params)
    covariance = chi_squared * np.linalg.inv(hessian)
    # R² Coeficient of Determination (dimensionless)
    r2 = 1 - residuals_sum_sq / np.linalg.norm(d - d.mean()) ** 2
    # signal-to-noise ratio in decibels (dB)
    SNR = 10 * np.log10(np.var(d, ddof=1) / np.var(residuals, ddof=1)) 
    return dipole_moment, covariance, r2, SNR


@numba.jit(nopython=True, parallel=True)
def _dipole_jacobian_fast(e, n, u, de, dn, du, jacobian):
    """
    This is the bit that runs the fast for-loops
    """
    constant = choclo.constants.VACUUM_MAGNETIC_PERMEABILITY / (4 * np.pi)
    for i in numba.prange(e.size):
        # Calculating the distance only once saves a lot of computation time
        distance = choclo.utils.distance_cartesian(
            e[i],
            n[i],
            u[i],
            de,
            dn,
            du,
        )
        # These are the second derivatives of 1/r
        jacobian[i, 0] = constant * choclo.point.kernel_eu(
            easting_p=e[i],
            northing_p=n[i],
            upward_p=u[i],
            easting_q=de,
            northing_q=dn,
            upward_q=du,
            distance=distance,
        )
        jacobian[i, 1] = constant * choclo.point.kernel_nu(
            easting_p=e[i],
            northing_p=n[i],
            upward_p=u[i],
            easting_q=de,
            northing_q=dn,
            upward_q=du,
            distance=distance,
        )
        jacobian[i, 2] = constant * choclo.point.kernel_uu(
            easting_p=e[i],
            northing_p=n[i],
            upward_p=u[i],
            easting_q=de,
            northing_q=dn,
            upward_q=du,
            distance=distance,
        )


def covariance_to_angle_std(dipole_moment, covariance):
    """
    Calculate the inc, dec, amp standard deviations from the covariance matrix
    """
    mx2, my2, mz2 = dipole_moment**2
    m = np.linalg.norm(dipole_moment)
    var_x, var_y, var_z = np.diag(covariance)
    sigma_inc = np.degrees(
        np.sqrt(
            (mx2 * mz2 * var_x + my2 * mz2 * var_y + (mx2 + my2) ** 2 * var_z)
            / ((mx2 + my2) * m**4)
        )
    )
    sigma_dec = np.degrees(np.sqrt((my2 * var_x + mx2 * var_y) / (mx2 + my2) ** 2))
    sigma_amp = np.sqrt((mx2 * var_x + my2 * var_y + mz2 * var_z) / m**2)
    return sigma_inc, sigma_dec, sigma_amp


def data_gradients(data):
    """
    Add the derivatives and total gradient amplitude
    """
    dx = data.differentiate("x")
    dy = data.differentiate("y")
    spacing = np.mean([np.abs(data.x[1] - data.x[0]), np.abs(data.y[1] - data.y[0])])
    # Need to set the exact same coordinates because the xrft inverse transform
    # creates slightly different ones because of round-off errors.
    data_up = hm.upward_continuation(data, spacing).assign_coords(
        dict(x=data.x, y=data.y)
    )
    data_down = hm.upward_continuation(data, -spacing).assign_coords(
        dict(x=data.x, y=data.y)
    )
    # Forward difference only to avoid downward continuation.
    dz = (data_up - data_down) / (2 * spacing)
    tga = np.sqrt(dx**2 + dy**2 + dz**2)
    tga.attrs = {"long_name": "total gradient amplitude", "units": "nT/µm"}
    dx.attrs = {"long_name": "x-derivative", "units": "nT/µm"}
    dy.attrs = {"long_name": "y-derivative", "units": "nT/µm"}
    dz.attrs = {"long_name": "z-derivative", "units": "nT/µm"}
    return xr.Dataset({"tga": tga, "x_deriv": dx, "y_deriv": dy, "z_deriv": dz})


def detect_anomalies(
    data, size_range, size_increment=2, nsizes=10, threshold=0.5, overlap=0.5, exclude_border=0
):
    """
    Run the blob detection and produce bounding boxes in data coordinates
    """
    min_sigma, max_sigma = [0.5 * i for i in size_range]
    spacing = np.mean([np.abs(data.x[1] - data.x[0]), np.abs(data.y[1] - data.y[0])])
    iy, ix, sigma_pix = skimage.feature.blob_log(
        data,
        min_sigma=min_sigma / spacing,
        max_sigma=max_sigma / spacing,
        threshold=threshold,
        num_sigma=nsizes,
        overlap=overlap,
        exclude_border= int(exclude_border / spacing),
    ).T
    blob_coords = (data.x.values[ix.astype("int")], data.y.values[iy.astype("int")])
    blob_sizes = sigma_pix * np.sqrt(2) * spacing * size_increment
    
## Experimentamos ordenar pelo tamanho das janelas, porém não deu muito certo!
#    order_vector = np.argsort(blob_sizes)[::-1]
    
#    windows = [
#        [blob_coords[0][i] - blob_sizes[i], blob_coords[0][i] + blob_sizes[i], 
#         blob_coords[1][i] - blob_sizes[i], blob_coords[1][i] + blob_sizes[i]]
#        for i in order_vector
#    ]
    
    windows = [
        [x - size, x + size, y - size, y + size]
        for size, x, y in zip(blob_sizes, *blob_coords)
    ]
    return windows

    
def euler_deconvolution(data, x_deriv, y_deriv, z_deriv):
    """
    Estimate the (x, y, z) position and base level by Euler Deconvolution
    """
    si = 3
    grids = xr.Dataset(
        dict(field=data, x_deriv=x_deriv, y_deriv=y_deriv, z_deriv=z_deriv)
    )
    table = vd.grid_to_table(grids)
    # Verde drops non-dimension coordinates so we have to add z back.
    # This is a bug in Verde.
    table["z"] = grids.z.values.ravel()
    n_data = table.shape[0]
    G = np.empty((n_data, 4))
    G[:, 0] = table.x_deriv
    G[:, 1] = table.y_deriv
    G[:, 2] = table.z_deriv
    G[:, 3] = si
    h = (
        table.x * table.x_deriv
        + table.y * table.y_deriv
        + table.z * table.z_deriv
        + si * table.field
    )
    p = np.linalg.solve(G.T @ G, G.T @ h)
    return p[:3], p[3]


def load_qdm(path):
    """
    Load QDM microscopy data in the Harvard group format.
    """
    contents = scipy.io.loadmat(path)
    # For some reason, the spacing is returned as an array with a single
    # value. That messes up operations below so get the only element out.
    spacing = contents['step'].ravel()[0] * METER_TO_MICROMETER
    bz = contents['Bz'] * TESLA_TO_NANOTESLA
    sensor_sample_distance = contents['h'] * METER_TO_MICROMETER
    x = np.arange(bz.shape[1]) * spacing
    y = np.arange(bz.shape[0]) * spacing
    z = np.full(bz.shape, sensor_sample_distance)
    data = vd.make_xarray_grid(
        (x, y, z), 
        bz, 
        data_names=["bz"], 
        dims=("y", "x"), 
        extra_coords_names="z",
    )
    data.x.attrs = {"units": "µm"}
    data.y.attrs = {"units": "µm"}
    data.z.attrs = {"long_name": "sensor sample distance", "units": "µm"}
    data.bz.attrs = {"long_name": "vertical magnetic field", "units": "nT"}
    data.attrs = {"file_name": str(path)}
    return data


def plot_dipole_moment(
    positions, dipole_moments, ax=None, vmin=None, vmax=None, cmap="plasma", cmap_norm=plt.Normalize, linewidth=1.5,
    add_colorbar=True, add_key=True, key_coords=(0.1, -0.1), **kwargs,
):
    """
    Plot the dipole moments on a map represented by vectors.
    
    Each dipole moment is represented by:
    
    * A vector with unit length, direction set by the declination, and color set by inclination.
    * A perpendicular line with the same color as the vector and with length set to the moment amplitude.
    """
    if ax is None:
        ax = plt.gca()
      
    inclination, declination, amplitude = vector_to_angles(dipole_moments)
    
    # Generate colors based on the amplitude values
    norm = cmap_norm(vmin=vmin, vmax=vmax)
    colors = plt.colormaps[cmap](norm(amplitude))
    
    positive_inc = inclination > 0
    negative_inc = ~positive_inc
    
    # Calculate the plot vector components
    scale = 90 - np.abs(inclination)
    u, v = scale * np.sin(np.radians(declination)), scale * np.cos(np.radians(declination))
    
    # Keyword arguments for the quiver plots
    args = dict(
        pivot="mid",
        width=0.005, 
        scale=15,
        headlength=1.5, 
        headaxislength=1.5,
        headwidth=2,
        minlength=1.5,
    )    
    args.update(kwargs)
    
    pos_quiver = ax.quiver(
        positions[:2][0][positive_inc], 
        positions[:2][1][positive_inc], 
        u[positive_inc], 
        v[positive_inc], 
        color=colors[positive_inc], 
        edgecolor='k', 
        linewidth=0, 
        **args,
    )    
    neg_quiver = ax.quiver(
        positions[:2][0][negative_inc], 
        positions[:2][1][negative_inc], 
        u[negative_inc], 
        v[negative_inc], 
        edgecolor=colors[negative_inc], 
        color="#ffffff00", 
        linewidth=linewidth,
        **args,
    )
    
    mappable = plt.cm.ScalarMappable(norm, cmap=cmap)    
    if add_colorbar:
        plt.colorbar(mappable, ax=ax, label="dipole moment (A.m²)")        
    if add_key:
        ax.quiverkey(pos_quiver, *key_coords, 90, label="$I = 0^\circ$", labelpos="S")
                     
    return mappable, pos_quiver, neg_quiver


def plot_stereonet(dipole_moments, ax=None, cmap="plasma", cmap_norm=plt.Normalize, vmin=None, vmax=None, label="", add_ticks=True, **kwargs):
    """
    Plot the dipole moments in a stereonet.
    
    The moment magnitude is set as the color. 
    Positive inclination symbols are solid, negative are just the outline.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='stereonet')

    ax.set_facecolor("#eeeeee") # color "#eeeeee"
    ax.grid(linestyle="-", color="#cccccc", linewidth=1)
    ax.set_longitude_grid_ends(90)
    if add_ticks:
        # Add ticks manually because mplstereonet doesn't work with subplots
        ax.set_azimuth_ticks([])
        ax.text(0.2, 1.65, '0°')
        ax.text(0.2, -1.8, '180°')
        ax.text(-2, -0.025, '270°')
        ax.text(1.65, -0.025, '90°')
    
    if label and not label.endswith(" "):
        label = label + " "
    
    inclination, declination, amplitude = vector_to_angles(dipole_moments)
    
    # Generate colors based on the amplitude values
    norm = cmap_norm(vmin=vmin, vmax=vmax)
    colors = plt.colormaps[cmap](norm(amplitude))
    
    positive_inc = inclination > 0
    ax.scatter(
        *mplstereonet.line(inclination[positive_inc], declination[positive_inc]),
        c=colors[positive_inc],
        label=f"{label}$I > 0$",
        edgecolors="#333333",
        **kwargs,
    )
    ax.scatter(
        *mplstereonet.line(-inclination[~positive_inc], declination[~positive_inc]),
        c="#ffffff00",
        label=f"{label}$I \le 0$",
        edgecolors=colors[~positive_inc],
        **kwargs,
    )

    mappable = plt.cm.ScalarMappable(norm, cmap=cmap)
    return mappable


def equal_area_projection(vectors):
    # norm = np.linalg.norm(vectors, axis=1)
    # vectors_unitary = vectors / norm[:, np.newaxis]
    vectors = np.asarray(vectors)
    vectors_unitary = np.empty_like(vectors)
    for i in range(vectors.shape[0]):
        vector = vectors[i]  
        norm = np.sqrt(np.sum(vector ** 2))
        vectors_unitary[i] = vector / norm
    inclinations, declinations, amplitudes = vector_to_angles(vectors)

    # XY_projected = np.zeros((len(vectors), 3))
    XY_projected = np.empty_like(vectors)
    for i, projected_vector in enumerate(vectors_unitary):
        r = np.sqrt(1 - np.abs(vectors_unitary[i, 2])) / np.sqrt(vectors_unitary[i, 0]**2 + vectors_unitary[i, 1]**2)
        XY_projected[i, 0] = r * vectors_unitary[i, 1]
        XY_projected[i, 1] = r * vectors_unitary[i, 0]
        XY_projected[i, 2] = amplitudes[i] if inclinations[i] >= 0 else -amplitudes[i]

    return XY_projected


class StereographicProjection:
    def __init__(self, vectors):
        self.vectors = vectors

    def plot(self, ax=None, cmap="inferno", cmap_norm=plt.Normalize, vmin=None, vmax=None, 
             label="", 
             # s=50, 
             add_ticks=True, draw_cross=True, add_radial_grid=True, 
             facecolor="#ffffff00", add_legend=False,
             **kwargs):
        """
        Draw the stereographic projection
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        # Add a face color
        background_circle = plt.Circle((0, 0), 1, color=facecolor, zorder=-2)
        ax.add_artist(background_circle)

        # Draw the great circle
        circle = plt.Circle((0, 0), 1, color='black', fill=False, zorder=3)
        ax.add_artist(circle)

        # Define the clipping area
        clip_path = plt.Circle((0, 0), 1, transform=ax.transData)

        if add_ticks:
            # Add ticks 
            ax.text(-0.025, 1.05, '0°')
            ax.text(-0.05, -1.075, '180°')
            ax.text(-1.15, -0.025, '270°')
            ax.text(1.025, -0.025, '90°')
        
        if label and not label.endswith(" "):
            label = label + " "
            # Draw the central cross if requested
            if draw_cross:
                hline = ax.axhline(y=0, color='black', zorder=-1)
                hline.set_clip_path(clip_path)
                vline = ax.axvline(x=0, color='black', zorder=-1)
                vline.set_clip_path(clip_path)


        # Draw the radial grid
        if add_radial_grid:
            for rad_dec in range(0, 360, 10):  # Grid lines every 10 degrees
                rad_inc = np.linspace(0, 90, 1000)  # Inclinations from 0 to 90 degrees
                # rad_int = np.ones_like(rad_inc)  # Constant amplitude for the grid lines

                # Generate the radial vectors
                radial_vector = angles_to_vector(rad_inc, rad_dec, 1)

                # Project the radial vectors
                radial_projected = equal_area_projection(radial_vector)

                # Plot each radial grid line
                ax.plot(radial_projected[:, 1], radial_projected[:, 0], color='gray', zorder=-2, lw=0.5)
            for circ_inc in range(0, 90, 10):
                circ_dec = np.linspace(0, 360, 1000)  # Inclinations from 0 to 90 degrees
                circ_int = np.ones_like(circ_dec)  # Constant amplitude for the grid lines

                # Generate the radial vectors
                circle_vector = angles_to_vector(circ_inc, circ_dec, circ_int)
                # Project the radial vectors
                circle_projected = equal_area_projection(circle_vector)

                # Plot each radial grid line
                ax.plot(circle_projected[:, 1], circle_projected[:, 0], color='gray', zorder=-2, lw=0.5)
        
        # Calculate the equal area projection
        XY_projected = equal_area_projection(self.vectors)

        # Generate colors based on the amplitude values
        norm = cmap_norm(vmin=vmin, vmax=vmax)
        colors = plt.colormaps[cmap](norm(abs(XY_projected[:, 2])))

        # Plotting the data
        positive_inc = XY_projected[:, 2] > 0
        scatter_pos = ax.scatter(XY_projected[:, 1][positive_inc], XY_projected[:, 0][positive_inc], 
                                 c=colors[positive_inc], edgecolors="#ffffff00", label=f"{label}$I > 0$",
                                **kwargs)
        scatter_pos.set_clip_path(clip_path)

        scatter_neg = ax.scatter(XY_projected[:, 1][~positive_inc], XY_projected[:, 0][~positive_inc],
                                 c="#ffffff00", edgecolors=colors[~positive_inc], label=rf"{label}$I \leq 0$",
                                **kwargs)
        scatter_neg.set_clip_path(clip_path)

        # Configure the axis
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if add_legend:
            ax.legend()
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Colormap (if necessary)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        return mappable


def generate_fourier_intensity(x, y, fit4a, fit4b):
    """
    Compute Fourier intensity based on power law fit.

    Parameters
    ----------
    x : array-like
        x-coordinate (should be centered at 0).
    y : array-like
        y-coordinate (should be centered at 0).
    fit4a : array-like, shape (3,)
        Coefficients for power-law fit.
    fit4b : array-like, shape (3,)
        Coefficients for power-law fit.

    Returns
    -------
    intensity : array-like
        Computed Fourier intensity.
    """

    transecta = np.exp(fit4a[0] * (x ** fit4a[1]) + fit4a[2], dtype=np.float64)
    transectb = fit4b[0] * (x ** fit4b[1]) + fit4b[2]

    return transecta * (y ** transectb) + 0


def qdm_noise(shape, std_dev=50, seed=None):
    """
    Generate a synthetic noise map based on Fourier intensity properties.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output noise map as (rows, cols).
    std_dev : float, optional
        Standard deviation of the noise distribution. Default is 50 nT.

    Returns
    -------
    Bz : array-like
        Generated noise map.
    """
    
    rng = np.random.default_rng(seed)

    realmap = np.zeros((shape[0] * 2, shape[1] * 2))
    outdimensions = (shape[0]+1, shape[1]+1)


    # Best-fit spectral power parameters
    cvalsa = [4.5666, -0.8939, -21.2259]
    cvalsb = [-1.5723, -0.5414, -0.0208]

    # Initialize output matrices
    realmapout1 = np.zeros((realmap.shape[0] // 2, realmap.shape[1] // 2))
    realmapout2 = np.zeros((realmap.shape[0] // 2, realmap.shape[1] // 2))

    end_j = realmapout1.shape[1]-1
    # Fill real part of Fourier map
    for i in range(realmapout1.shape[0]):
        for j in range(realmapout1.shape[1]):
            if i == 0 or j == 0:
                continue
            sigma = np.sqrt(generate_fourier_intensity(j, i, cvalsa, cvalsb),  dtype=np.float64)  # Shift index
            realmapout1[i, end_j -j + 1] = rng.normal(0, sigma)
            realmapout2[i, j] = rng.normal(0, sigma)

    realmapout = np.hstack((realmapout1, realmapout2))

    # Initialize imaginary part of Fourier map
    imagmapout1 = np.zeros_like(realmapout1)
    imagmapout2 = np.zeros_like(realmapout2)

    for i in range(imagmapout1.shape[0]):
        for j in range(imagmapout1.shape[1]):
            if i == 0 or j == 0:
                continue
            sigma = np.sqrt(generate_fourier_intensity(j, i, cvalsa, cvalsb),  dtype=np.float64)  # Shift index
            imagmapout1[i, end_j -j + 1] = rng.normal(0, sigma)
            imagmapout2[i, j] = rng.normal(0, sigma)

    imagmapout = np.hstack((imagmapout1, imagmapout2))

    # Construct the complex Fourier map
    halfmapout = realmapout + 1j * imagmapout

    # Construct full Fourier map with symmetry
    fullmapout = np.zeros((halfmapout.shape[0] * 2, halfmapout.shape[1]), dtype=complex)
    fullmapout[-halfmapout.shape[0]:, :] = halfmapout
    fullmapout[:-halfmapout.shape[0], :] = np.rot90(halfmapout, 2)


    # Apply inverse Fourier transform
    temp2 = ifft(fftshift(np.sqrt(fullmapout)))


    # Extract final noise map
    rows = [
        round(temp2.shape[0] / 2 - outdimensions[0] / 2),
        round(temp2.shape[0] / 2 - outdimensions[0] / 2 + outdimensions[0] - 1)
    ]
    cols = [
        round(temp2.shape[1] / 2 - outdimensions[1] / 2),
        round(temp2.shape[1] / 2 - outdimensions[1] / 2 + outdimensions[1] - 1)
    ]

    outBz = np.real(temp2[rows[0]:rows[1], cols[0]:cols[1]])

    return (outBz/outBz.max())*3*std_dev


