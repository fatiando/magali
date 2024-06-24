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

    data.bz.plot(cmap="RdBu_r", vmin=-1000, vmax=1000)
