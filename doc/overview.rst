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

    import pooch
    import matplotlib.pyplot as plt


Load some data:

.. jupyter-execute::

    path = pooch.retrieve(
        "doi:10.6084/m9.figshare.22965200.v1/Bz_uc0.mat",
        known_hash="md5:268bd3af5e350188d239ff8bd0a88227",
    )
    print(path)

.. jupyter-execute::

    data = mg.read_qdm_harvard(path)
    data

.. jupyter-execute::

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8), layout="constrained")
    scale = 2500
    data.plot.imshow(ax=ax, cmap="RdBu_r", vmin=-scale, vmax=scale)
    ax.set_aspect("equal")
    plt.show()