![header](https://raw.githubusercontent.com/marcoalopez/PyRockWave/main/img/header.jpg)

This project is maintained by [Marco A. Lopez-Sanchez](https://marcoalopez.github.io/) - Last update: 2026-07-10

## **What is PyRockWave?**

**PyRockWave** is a free, open-source Python tool that reads single-crystal elasticity databases and calculates the elastic properties of Earth materials. Its submodules can compute isotropic seismic properties,  volume- and CPO-weighted averages, phase and group seismic properties from the Christoffel equation, analytical anisotropy models, reflectivity and effective stiffness/compliance tensors for layered media (Schoenberg & Muir approach), and ultrasonic signal processing. Further tools visualize seismic properties, convert between spherical and Cartesian coordinates, generate S2 grids, and simplify working with tensors.

**Why PyRockWave?**

TODO

PyRockWave is modular: it exposes plain Python functions so you can develop with it and integrate it into other tools. Worked examples in Jupyter notebooks keep workflows easy to follow and reproducible.



> [!WARNING]
> PyRockWave is in the beta stage. Its functionality may change and break backward compatibility. Parts of the code are not well tested yet, and there is no installation guide or detailed documentation. However, the code will always be free and open source. You are welcome to explore and use it at your own risk. View repository on GitHub: https://github.com/marcoalopez/PyRockWave

## **Requirements & Python installation**

PyRockWave requires **Python 3.10 or later** and depends on NumPy, SciPy, Pandas and Matplotlib, which are installed automatically. Please note that PyRockWave requires the NumPy 2 generation of the scientific Python stack (mid-2024 or later). We strongly recommend that you install PyRockWave in a dedicated environment or workspace to avoid compatibility issues with general-purpose Python environments. The installation instructions below are based on this assumption.

Available soon (I'm working on it! 😊)

## **PyRockWave wiki**

Available soon (I'm working on it! 😊)

## **Jupyter Notebooks library**

Available soon (I'm working on it! 😊)

- Accessing elastic databases and estimating isotropic seismic properties with the ``elastic_tensor`` submodule

- Demonstration of the functionality of the ``christoffel`` submodule

- Demonstration of the functionality of the ``layered_media`` submodule

- Demonstration of the functionality of the ``averaging _schemes`` submodule

- Demonstration of the functionality of the ``anisotropic_models`` submodule

- [Demonstration of the functionality of the ``coordinates`` module](https://github.com/marcoalopez/PyRockWave/blob/main/notebooks/example_coordinates.ipynb)

- [Demonstration of the functionality of the ``tensor_tools`` module](https://github.com/marcoalopez/PyRockWave/blob/main/notebooks/example_tensor_tools.ipynb)


## **How to contribute to this project?**

The GitHub website hosting the project provides several options (you will need a GitHub account, it’s free!):

- Open a [discussion](https://github.com/marcoalopez/PyRockWave/discussions): This is a place to:
  - Ask questions.
  - Request features or share ideas with the developers.
- Open an [issue](https://github.com/marcoalopez/PyRockWave/issues): This is a place to report and track bugs.
- Create a [pull request](https://github.com/marcoalopez/PyRockWave/pulls): Submit a fix or new feature for a notebook, and a developer will review it and merge it.

## **Funding**

Funding from the Government of the Principality of Asturias and the Foundation for the Promotion of Applied Research in Asturias (FICYT) (grant: SV-PA-21-AYUD/2021/57163) under the Asturias Plan for Science, Technology and Innovation (PCTI-Asturias) 2018-2022 seeded this project. I am currently seeking further funding to complete it.

---
*Copyright © 2023-present Marco A. Lopez-Sanchez*  

> [!NOTE]
> _The information on this website and in the script documentation is provided without any warranty of any kind, either expressed or implied, and may include technical inaccuracies or typographical errors. The author reserves the right to make changes or improvements to the content of this website and the script documentation at any time without notice. This website and its documentation are not responsible for the content of external links. Notebook content is licensed under [Creative Commons Attribution license CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) and code is licensed under GNU General Public License v3 (https://www.gnu.org/licenses/gpl-3.0.en.html) unless otherwise noted._

_Hosted on GitHub Pages — This website was created with [Typora](https://typora.io/)_