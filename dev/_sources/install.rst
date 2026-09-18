.. _install:

Installing
==========

There are different ways to install magali:

.. tab-set::

    .. tab-item:: pip

        Using the `pip package manager <https://pypi.org/project/pip/>`__:

        .. code:: bash

            pip install magali

    .. tab-item:: conda/mamba

        Using the `conda package manager <https://conda.io/>`__ (or ``mamba``)
        that comes with the Anaconda/Miniconda distribution:

        .. code:: bash

            conda install magali --channel conda-forge

    .. tab-item:: Developement version

        You can use ``pip`` to install the latest **unreleased** version from
        GitHub (**not recommended** in most situations):

        .. code:: bash

            python -m pip install --upgrade git+https://github.com/compgeolab/magali

.. note::

   The commands above should be executed in a terminal. On Windows, use the
   ``cmd.exe`` or the "Anaconda Prompt" app if you’re using Anaconda.


Which Python?
-------------

You'll need **Python >= 3.9**.
See :ref:`python-versions` if you require support for older versions.

Dependencies
------------

These required dependencies should be installed automatically when you install
Magali with ``pip`` or ``conda``:

* `numpy <https://www.numpy.org/>`__
* `scipy <https://scipy.org/>`__
* `numba <https://numba.pydata.org/>`__
* `scikit-image <https://scikit-image.org/>`__
* `xarray <https://xarray.dev/>`__
* `harmonica <https://www.fatiando.org/harmonica/>`__
* `verde <https://www.fatiando.org/verde/>`__
* `choclo <https://www.fatiando.org/choclo/>`__

See :ref:`dependency-versions` for the our policy of oldest supported versions
of each dependency.
