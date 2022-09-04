# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

PATH_ROOT = Path(__file__).parent.resolve()

version = "0.1.0a0"
sha = "Unknown"
package_name = "huo"

try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PATH_ROOT).decode("ascii").strip()
except Exception:
    pass

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version += "+" + sha[:7]


def write_version_file():
    version_path = PATH_ROOT / package_name / "version.py"
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")


def get_long_description():
    # Get the long description from the README file
    description = (PATH_ROOT / "README.md").read_text(encoding="utf-8")
    # (TODO): replace relative repository path to absolute link to the release
    return description


def load_requirements(path_dir=PATH_ROOT, file_name="requirements.txt", comment_char="#"):
    with open(path_dir / file_name, "r", encoding="utf-8", errors="ignore") as file:
        lines = [ln.rstrip() for ln in file.readlines() if not ln.startswith("#")]
    reqs = []
    for ln in lines:
        if comment_char in ln:  # filer all comments
            ln = ln[: ln.index(comment_char)].strip()
        if ln.startswith("http"):  # skip directly installed dependencies
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    write_version_file()

    setup(
        name=package_name,
        version=version,
        description="Huo (wip)",
        author="Zhiqiang Wang",
        author_email="zhiqwang@outlook.com",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/zhiqwang/huo",
        license="GPL-3.0",
        packages=find_packages(exclude=["test"]),
        zip_safe=False,
        classifiers=[
            # Operation system
            "Operating System :: OS Independent",
            # How mature is this project? Common values are
            #   3 - Alpha, 4 - Beta, 5 - Production/Stable
            "Development Status :: 3 - Alpha",
            # Indicate who your project is intended for
            "Intended Audience :: Developers",
            # Topics
            "Topic :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Recognition",
            # Pick your license as you wish
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            # Specify the Python versions you support here.
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        install_requires=load_requirements(),
        # This field adds keywords for your project which will appear on the
        # project page. What does your project relate to?
        #
        # Note that this is a list of additional keywords, separated
        # by commas, to be used to assist searching for the distribution in a
        # larger catalog.
        keywords="scientific-computing, machine-learning, deep-learning",
        # Specify which Python versions you support. In contrast to the
        # 'Programming Language' classifiers above, 'pip install' will check this
        # and refuse to install the project if the version does not match. See
        # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
        python_requires=">=3.7",
        # List additional URLs that are relevant to your project as a dict.
        #
        # This field corresponds to the "Project-URL" metadata fields:
        # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
        #
        # Examples listed include a pattern for specifying where the package tracks
        # issues, where the source is hosted, where to say thanks to the package
        # maintainers, and where to support the project financially. The key is
        # what's used to render the link text on PyPI.
        project_urls={  # Optional
            "Bug Reports": "https://github.com/zhiqwang/huo/issues",
            "Funding": "https://zhiqwang.com",
            "Source": "https://github.com/zhiqwang/huo/",
        },
    )
