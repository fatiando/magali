.. _overview:

Overview
========

**Magali** is a Python library for the forward modeling, inversion, and analysis
of magnetic microscopy data,with a particular focus on paleomagnetism and rock
magnetism applications. It provides tools to infer the magnetic properties of
individual mineral grains from high-resolution magnetic microscopy images, such 
as those acquired by quantum diamond microscopes (QDMs).

Its main goals are:

- Offer robust and efficient implementations of dipole modeling and inversion 
  methods tailored for magnetic microscopy data.
- Enable the semi-automatic detection, localization, and characterization of
  hundreds to thousands of magnetic sources in a single microscopy image.
- Support reproducible research and community-driven development by providing
  well-documented, open-source code.
- Facilitate methodological innovation by offering a modular and extensible
  platform for the development and testing of new techniques.

The library
-----------
All functionality in magali is available from the base namespace of the 
:mod:`magali` package. This means that you can access all of them with a 
single import:

.. jupyter-execute::

    # magali is usually imported as mg
    import magali as mg


.. seealso::

    Checkout the :ref:`api` for a comprehensive list of the available functions
    and classes in Magali.