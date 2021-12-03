FAQ
##############

.. toctree::
    :maxdepth: 2

Carla installation problem
============================

Q1: No `easy_install`
-----------------------

:A1:
    ``easy_install`` is used in old version of ``setuptools``, and may not be added in conda/pyenv by
    default. You may need to install by yourself.

    .. code:: bash

        pip install setuptools==50

Q2: `carla.egg` installation error or can not import
--------------------------------------------------------------

:A2:
    This is usually because the ``easy_install`` command you have run (linked to) is not the one 
    installed in your environment.
    You can manually add it to your ``PATH`` or simply find in
    the ``./bin`` folder in your environment path and directly use it.

Q3: Error with 'libpng', 'libjpeg'
--------------------------------------

:A3:
    Carla python API needs library of ``png`` and ``jpeg`` that may not be installed in your environment.
    Install them manually if error shows.

    png:

    .. code:: bash

        # in system
        sudo apt-get install libpng16
        # in conda env
        conda install libpng -c anaconda 

    jpeg:

    .. code:: bash

        # in system
        sudo apt-get install libjpeg-dev 
        # in conda env
        conda install jpeg=8d -c conda-forge
    
Q4: Error with 'ALSA lib confxxx'
-----------------------------------

:A4:
    This error shows because Carla needs a sound card to run simulation.
    If this makes no sense to anything, just ignore it. Or you can fix it by adding 
    ``pcm.!default { type plug slave.pcm "null" }``
    into a new line of ``/etc/asound.conf`` file in your system.

Q5: Error import `shapely`
------------------------------

:A5:
    This often happens with showing the following error log.

    .. code::

        OSError: [WinError 126] The specified module could not be found

        OSError: Could not find library geos_c

    You can solve it by install shapely again or install geos manually.

    In Windows:
        Reinstall shapely with geos from published wheels at
        https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely.

    In Ubuntu:
        .. code:: bash

            sudo apt-get install libgeos-dev
