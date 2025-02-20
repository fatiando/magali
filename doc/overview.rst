.. _overview:

Overview
========

The library
-----------

All functionality in magali is available from the base namespace of the
:mod:`magali` package. This means that you can access all of them with a
single import:

.. jupyter-execute::

    # magali is usually imported as mg
    import magali as mg

    import ensaio
    import matplotlib.pyplot as plt


Load some data:

.. jupyter-execute::

    fname = ensaio.fetch_morroco_speleothem_qdm(version=1, file_format="netcdf")
    print(fname)

.. jupyter-execute::

    data = mg.read_qdm_harvard(fname)
    data

.. jupyter-execute::

    data.bz.plot(cmap="RdBu_r", vmin=-1000, vmax=1000)

