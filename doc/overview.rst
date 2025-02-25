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


    fname = ensaio.fetch_morroco_speleothem_qdm(version=1, file_format="matlab")
    print(fname)

.. jupyter-execute::

    data = mg.read_qdm_harvard(fname)
    data

.. jupyter-execute::

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8), layout="constrained")
    scale = 2500
    data.plot.imshow(ax=ax, cmap="RdBu_r", vmin=-scale, vmax=scale)
    ax.set_aspect("equal")
    plt.show()