Loading a Dataset
=================

Loading data is a fundamental step in any data analysis or modeling workflow.
The raw data must be accessed in a consistent, reliable, and reproducible manner.
In this example, we use `Ensaio <https://github.com/fatiando/ensaio>`_ to fetch a dataset containing microscopy measurements of a speleothem, ensuring both accessibility and data integrity.

.. admonition:: What is Ensaio?
    :class: seealso

    `Ensaio <https://github.com/fatiando/ensaio>`_ serves as a reliable source for sharing reproducible research data in geophysics, providing persistent URLs
    and metadata for each dataset. The datasets in Ensaio are stored on platforms like Figshare or Zenodo,
    and each one includes a unique `SHA256` has to ensure file integrity.

To load a dataset, the data is fetched using `Ensaio <https://github.com/fatiando/ensaio>`_ 
as and read with Magali as an `DataArray`:

.. jupyter-execute::

    import magali as mg
    import ensaio

    fname = ensaio.fetch_morroco_speleothem_qdm(version=1, file_format="matlab")
    bz = mg.read_qdm_harvard(fname)

.. admonition:: What is DataArray?
    :class: seealso

    A DataArray is the core data structure in xarray, similar to a labeled N-dimensional `NumPy` array. It includes:

    - **data:** the actual array values
    - **dims:** names of dimensions
    - **coords:** coordinate labels for each dimension
    - **attrs:** metadata

Inspecting the Data
-------------------

To understand its structure and content, it’s helpful to inspect its dimensions, coordinates, and metadata:

.. jupyter-execute::

    print(bz)           # Summary view
     
As you can see, the DataArray corresponds to the vertical magnetic field, 
which was measured in `nT` and has x, y and z coordinates.

We can also check the coordinates units:

.. jupyter-execute::

    print(bz.x.units)
    print(bz.y.units)
    print(bz.z.units)

Plotting the Data
-------------------

Finally, the data is plotted:

.. jupyter-execute::

    bz.plot.pcolormesh(cmap="seismic", vmin=-5000, vmax=5000)
