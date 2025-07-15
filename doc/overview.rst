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
  well-documented open-source code.
- Facilitate methodological innovation by offering a modular and extensible
  platform for the development and testing of new techniques.

Conventions
-----------

Magali adopts the following conventions across its API:

- All physical quantities follow the
  `International System of Units (SI) <https://en.wikipedia.org/wiki/International_System_of_Units>`__
  , with a few exceptions:

  - **Magnetic fields** are expressed in **nanoTesla (nT)**.
  - **Magnetic moments** are expressed in **Ampere square meters (A·m²)**.

- A **right-handed Cartesian coordinate system** is used throughout, defined as:

  - The **x-axis** runs horizontally across the image and is analogous to
    **easting**.
  - The **y-axis** lies in the image plane, perpendicular to x, and is analogous
    to **northing**.
  - The **z-axis** points **upward**, away from the sample surface (i.e., out of
    the image plane).

This orientation follows the right-hand rule: if you point your right-hand 
fingers along the x-axis and curl them toward the y-axis, your thumb will point 
in the direction of the z-axis.

.. note::

   While the x and y axes may resemble **easting** and **northing**, microscopy
   data lacks a globally georeferenced frame. Instead, axis orientation depends
   on the acquisition setup or sample mounting. To ensure correct modeling and
   interpretation, always define and maintain a consistent axis convention 
   throughout your workflow.


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