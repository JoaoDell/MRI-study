# How to develop a library using pyproject.toml

To have a pyproject.toml working, some things are necessary:

1. "build-system"

2. "project"

3. Build backend details

## [build-system]

The first one will define the build backend that will be used. The one used for this project is hatchling, however, there are several others available online.

An example would be:

    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"
    
## [project]

The second one will contain the project's metadata, like name, version, description, etc. The most basic project metadata would be:

    [project]
    name = "pyMRI"
    version = "2024.04.09"

## Build backend details

The build backend details are basically the required instructions your building backend needs to operate. The hatchling backend, for example, requires:

    [tool.hatch.build.targets.wheel]
    packages = ["path/to/package"]

So it can find the package it will install.

## Installing it in developer mode

To install a local package in development mode, use pip with the following command:

    pip install -e path/to/pyproject.toml/folder

Or simply type:

    pip install -e .

Inside the pyproject.toml folder.

## References

1. Writing your pyproject.toml [https://packaging.python.org/en/latest/guides/writing-pyproject-toml/](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

2. FURY's pyproject.toml [https://github.com/JoaoDell/fury/blob/feat/api-kde/pyproject.toml](https://github.com/JoaoDell/fury/blob/feat/api-kde/pyproject.toml)

3. Packaging Python Projects [https://packaging.python.org/en/latest/tutorials/packaging-projects/](https://packaging.python.org/en/latest/tutorials/packaging-projects/)


