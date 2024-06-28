![header](https://raw.githubusercontent.com/marcoalopez/PyRockWave/main/img/header.jpg)

This project is maintained by [Marco A. Lopez-Sanchez](https://marcoalopez.github.io/) - Last update: 2024-06-28 

## What is PyRockWave?

PyRockWave is a Python-based tool for reading single-crystal elasticity databases and modeling the elastic properties of earth materials (e.g. seismic wave propagation and anisotropy in heterogeneous media). It allows you to create models of the physical properties of rocks or rock units and to simulate and illustrate how seismic waves propagate through them. The code is designed in a modular way to facilitate its development and the interaction with other codes.

View repository on GitHub: https://github.com/marcoalopez/PyRockWave

> [!CAUTION]
> PyRockWave is still under development (alpha state), so the functionality may be subject to major changes that break backwards compatibility, some of the code is untested, and there is no installation guide or specific documentation on how to use it yet. In any case, the code is (and always will be) open source and free, so even though there is no official release and very little documentation, you can have fun with it at your own risk.

## Requirements & Python installation

TODO

## How to use it?

- [Interacting with the mineral elastic data](https://github.com/marcoalopez/PyRockWave/blob/main/src/example_database.ipynb)

- [The ElasticProps class explained](https://github.com/marcoalopez/PyRockWave/blob/main/src/ElasticTensor_explained.ipynb)

- [Demonstration of the functionality of the ``coordinates`` module](https://github.com/marcoalopez/PyRockWave/blob/main/src/example_coordinates.ipynb)

- [Demonstration of the functionality of the ``polar_models`` module](https://github.com/marcoalopez/PyRockWave/blob/main/src/example_polar.ipynb)

- [Demonstration of the functionality of the ``orthorhombic_models`` module](https://github.com/marcoalopez/PyRockWave/blob/main/src/example_orthotropic.ipynb)

- Demonstration of the functionality of the ``christoffel`` module (TODO)

- Demonstration of the functionality of the ``layered_media`` module (TODO)

- Demonstration of the functionality of the ``tensor_tools`` module (TODO)

## How to contribute to this project?

The GitHub website hosting the project provides several options (you will need a GitHub account, it’s free!):

- Open a [discussion](https://github.com/marcoalopez/PyRockWave/discussions): This is a place to:
  - Ask questions you are wondering about.
  - Requests for specific features or share ideas.
  - Interact with the developers (still just me).
- Open and [issue](https://github.com/marcoalopez/PyRockWave/issues): This is a place to report or track bugs.
- Create a [pull request](https://github.com/marcoalopez/PyRockWave/pulls): You modified, corrected or added a feature to one of the notebooks and send it for one of the developers to review it and add it to the main page.

## Funding

The seed of these codes has been made possible thanks to funding from the Government of the Principality of Asturias and the Foundation for the Promotion of Applied Research in Asturias (FICYT) (grant SV-PA-21-AYUD/2021/57163) under the Asturias Plan for Science, Technology and Innovation 2018-2022 (PCTI-Asturias). 

---
*Copyright © 2024 Marco A. Lopez-Sanchez*  

> [!WARNING]
>  _The information on this website and in the script documentation is provided without any warranty of any kind, either expressed or implied, and may include technical inaccuracies or typographical errors; the author reserves the right to make changes or improvements to the content of this website and the script documentation at any time without notice. This website and its documentation are not responsible for the content of external links. Notebook contents under [Creative Commons Attribution license CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) and codes under GNU General Public License v3 (https://www.gnu.org/licenses/gpl-3.0.en.html) unless otherwise indicated._

_Hosted on GitHub Pages — This website was created with [Typora](https://typora.io/)_