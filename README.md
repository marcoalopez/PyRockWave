![header](https://raw.githubusercontent.com/marcoalopez/PyRockWave/main/img/header.jpg)

This project is maintained by [Marco A. Lopez-Sanchez](https://marcoalopez.github.io/) — Last updated: 12 July 2026

## **What is PyRockWave?**

**PyRockWave** is a free, open-source Python tool that reads single-crystal elasticity databases and calculates the elastic properties of Earth materials. Its submodules compute isotropic seismic properties, volume- and CPO-weighted averages, phase and group seismic properties from the Christoffel equation, seismic velocities from analytical anisotropy models, and reflectivity and effective stiffness/compliance tensors for layered media (Schoenberg & Muir approach); a dedicated submodule processes ultrasonic signals. Further tools visualise seismic properties, convert between spherical and Cartesian coordinates, generate S2 grids, and simplify working with tensors.

## **Why PyRockWave?**

- **One tool for the whole workflow**: access elastic databases, import texture/CPO data from various sources, model elastic properties and anisotropy, and produce publication-ready plots — all without leaving Python.

- **Modularity**: PyRockWave exposes plain Python functions, so you can build on it and integrate it into other tools. All functions are documented.

- **Interactivity**: worked examples in Jupyter notebooks make workflows easy to follow and to reproduce.

- **Few, solid dependencies**: PyRockWave is built on four widely used scientific Python libraries — [NumPy](https://numpy.org/) and Pandas for numerical and tabular data, [SciPy](https://scipy.org/) for SO(3) rotations and optimisation, and Matplotlib for plotting — and nothing else.


> [!WARNING]
> PyRockWave is in beta and not yet released. Its functionality may change and break backward compatibility. There is, as yet, no installation guide or detailed documentation. You are welcome to explore it at https://github.com/marcoalopez/PyRockWave and use it at your own risk.


## **Requirements and installation**

PyRockWave requires **Python 3.10 or later** and depends on NumPy, SciPy, Pandas and Matplotlib, which are installed automatically. PyRockWave requires the NumPy 2 generation of the scientific Python stack (mid-2024 or later). I strongly recommend that you install PyRockWave in a dedicated environment or workspace to avoid compatibility issues with general-purpose Python environments. The instructions below assume this.

In preparation.

## **PyRockWave wiki**

In preparation.

## **Example notebooks**

In preparation.

- ``elastic_tensor`` — accessing elastic databases and estimating isotropic seismic properties

- ``christoffel`` — phase and group seismic properties from the Christoffel equation

- ``layered_media`` — reflectivity and effective stiffness/compliance tensors for layered media

- ``averaging_schemes`` — volume- and CPO-weighted averages

- ``anisotropic_models`` — seismic velocities from analytical anisotropy models

- [``coordinates`` — spherical/Cartesian conversions and S2 grids](https://github.com/marcoalopez/PyRockWave/blob/main/notebooks/example_coordinates.ipynb)

- [``tensor_tools`` — tensor manipulation helpers](https://github.com/marcoalopez/PyRockWave/blob/main/notebooks/example_tensor_tools.ipynb)


## **How to contribute**

You can contribute through GitHub (a free account is required):

- Open a [discussion](https://github.com/marcoalopez/PyRockWave/discussions) to ask questions, request features, or share ideas.
- Open an [issue](https://github.com/marcoalopez/PyRockWave/issues) to report a bug.
- Create a [pull request](https://github.com/marcoalopez/PyRockWave/pulls) to submit a fix or improvement; the maintainer will review it before merging.

## **Funding**

A grant from the Government of the Principality of Asturias and the Foundation for the Promotion of Applied Research in Asturias (FICYT), under the Asturias Plan for Science, Technology and Innovation (PCTI-Asturias) 2018–2022 (SV-PA-21-AYUD/2021/57163), seeded this project. I am seeking further funding to complete it.

---
*Copyright © 2023–present Marco A. Lopez-Sanchez*  

> [!NOTE]
> _The information on this website and in the script documentation is provided without any warranty of any kind, either express or implied, and may include technical inaccuracies or typographical errors. The author reserves the right to make changes or improvements to the content of this website and the script documentation at any time without notice. The author is not responsible for the content of external links. Notebook content is licensed under the [Creative Commons Attribution licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) and code is licensed under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html) unless otherwise noted._

_Hosted on GitHub Pages. This website was created with [Typora](https://typora.io/)_
