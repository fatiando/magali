.. _souza_junior_method:

Overview
========

The methodology comprises three primary steps:

- **Step 1 – Source Detection**  
  Initial detection of potential magnetic sources from the data.

- **Step 2 – Iterative Processing (per window)**  
  For each detected anomaly:

  - **Data isolation**: Magnetic data within the window are extracted.
  - **Euler deconvolution**: Source location is estimated.
  - **Linear inversion**: Dipole moment is estimated assuming a fixed position.
  - **Non-linear inversion**: Position and moment are refined using the Nelder-Mead method.
  - **Signal removal**: Dipole response is forward modeled and subtracted from the dataset.

- **Step 3 – Detection on Residual Data**  
  Steps 1 and 2 are reapplied to the updated (stripped) dataset to identify and model additional sources.

.. jupyter-execute::

    import harmonica as hm
    import matplotlib.pyplot as plt
    import numpy as np
    import skimage.exposure
    import matplotlib.patches
    import magali as mg

Synthetic Data
==============

A synthetic model is generated to validate the methodology.

.. jupyter-execute::

    sensor_sample_distance = 5.0  # µm
    region = [0, 2000, 0, 2000]  # µm
    spacing = 2  # µm

    true_inclination = 30
    true_declination = 40
    true_dispersion_angle = 5

    size = 100

    directions_inclination, directions_declination = mg.random_directions(
        true_inclination,
        true_declination,
        true_dispersion_angle,
        size=size,
        random_state=5,
    )

    dipoles_amplitude = abs(np.random.normal(0, 100, size)) * 1.0e-14

    dipole_coordinates = (
        np.concatenate([np.random.randint(30, 1970, size), [1250, 1300, 500]]),  # µm
        np.concatenate([np.random.randint(30, 1970, size), [500, 1750, 1000]]),  # µm
        np.concatenate([np.random.randint(-20, -1, size), [-15, -15, -30]]),  # µm
    )
    dipole_moments = hm.magnetic_angles_to_vec(
        inclination=np.concatenate([directions_inclination, [10, -10, -5]]),
        declination=np.concatenate([directions_declination, [10, 170, 190]]),
        intensity=np.concatenate([dipoles_amplitude, [5e-11, 5e-11, 5e-11]]),
    )

    data = mg.dipole_bz_grid(
        region, spacing, sensor_sample_distance, dipole_coordinates, dipole_moments
    )
    data.plot.pcolormesh(cmap="seismic", vmin=-5000, vmax=5000)

Source Detection
================

The source detection process involves several signal enhancement and segmentation steps:

- **Upward continuation**: High-frequency noise is reduced.
- **Total Gradient Amplitude (TGA)**: Signals near magnetic sources are enhanced.
- **Contrast stretching**: Weaker signals are highlighted through rescaling.
- **Laplacian of Gaussian (LoG) segmentation**: Anomalies are identified by detecting "blobs".
- **Ranking**: Windows are ranked by signal strength to prioritize processing.

Upward Continuation
===================

Upward continuation suppresses small-scale noise while preserving larger magnetic anomalies.  
A minimal upward continuation height is selected to retain most of the original signal.

.. jupyter-execute::

    height_difference = 5
    data_up = (
        hm.upward_continuation(data, height_difference)
        .assign_attrs(data.attrs)
        .assign_coords(x=data.x, y=data.y)
        .assign_coords(z=data.z + height_difference)
    )
    data_up.plot.pcolormesh(cmap="seismic", vmin=-50000, vmax=50000)

Total Gradient Amplitude (TGA)
==============================

The TGA acts as a high-pass filter, emphasizing regions near magnetic sources.

.. jupyter-execute::

    data_tga = mg.total_gradient_amplitude_grid(data_up)
    data_tga.plot.pcolormesh(cmap="seismic")

Contrast Stretching
===================

Contrast stretching is applied to enhance both weak and strong signals, based on the 1st and 99th percentiles.

.. jupyter-execute::

    data_stretched = skimage.exposure.rescale_intensity(
        data_tga, 
        in_range=tuple(np.percentile(data_tga, (1, 99))),
    )
    data_stretched.plot.pcolormesh(cmap="gray_r")

Laplacian of Gaussian (LoG) Segmentation
========================================

Anomalies are detected using the LoG method, identifying "blobs" corresponding to individual particles.  
The detection function is configured to use parameters in physical units (µm) and returns bounding boxes (`x_min`, `x_max`, `y_min`, `y_max`) in the same units.

.. jupyter-execute::

    windows = mg.detect_anomalies(
        data_stretched,
        size_range=[25, 50],  # Expected size range of anomalies (µm)
        size_multiplier=2,
        num_scales=10,
        detection_threshold=0.01,
        overlap_ratio=0.5,
        border_exclusion=1,
    )
