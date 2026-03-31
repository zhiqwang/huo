# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

"""Thin wrapper — delegates to ``huo.cli.main``.

Prefer using the installed ``huo-art`` command instead::

    huo-art ./data/sinogram.npy -o reconstruction.npy
"""

from huo.cli import main

if __name__ == "__main__":
    main()
