![header](https://raw.githubusercontent.com/marcoalopez/PyRockWave/main/img/header.jpg)

This project is maintained by [Marco A. Lopez-Sanchez](https://marcoalopez.github.io/) - Last update: 2025-07-15

## What is PyRockWave?

**PyRockWave** is a free, open-source Python tool for reading single-crystal elastic databases and modeling the elastic properties of Earth materials. It computes and visualizes physical properties of minerals, rocks, and layered rock units using various averaging models. The tool emphasizes seismic anisotropy —the directional variation of seismic wave velocities within materials— which provides insights into mineral orientation, stress fields, and flow patterns in Earth's mantle and crust.

Designed with modularity in mind using Python functions, PyRockWave facilitates development and integration with other tools. Example applications, built using Jupyter notebooks, ensure ease of use and reproducible workflows.

View repository on GitHub: https://github.com/marcoalopez/PyRockWave

> [!CAUTION]
> PyRockWave is still under development (alpha state). Functionality may change significantly, potentially breaking backward compatibility. Most of the code is undertested, and there is no installation guide or detailed documentation yet. However, the code is free and open source and always will be. Although there is no official release or comprehensive documentation, you are welcome to explore and use the software at your own risk..

## Requirements & Python installation

TODO

## Examples

- [Interacting with the mineral elastic data](https://github.com/marcoalopez/PyRockWave/blob/main/src/example_database.ipynb)

- [The ElasticProps class explained](https://github.com/marcoalopez/PyRockWave/blob/main/src/ElasticTensor_explained.ipynb)

- [Demonstration of the functionality of the ``coordinates`` module](https://github.com/marcoalopez/PyRockWave/blob/main/src/example_coordinates.ipynb)

- [Demonstration of the functionality of the ``tensor_tools`` module](https://github.com/marcoalopez/PyRockWave/blob/main/src/example_tensor_tools.ipynb)

- [Demonstration of the functionality of the ``anisotropic_models`` module](https://github.com/marcoalopez/PyRockWave/blob/main/src/example_anisotropic_models.ipynb)

- Demonstration of the functionality of the ``christoffel`` module (I'm working on it! 😊)

- Demonstration of the functionality of the ``layered_media`` module (I'm working on it! 😊)

## How to contribute to this project?

The GitHub website hosting the project provides several options (you will need a GitHub account, it’s free!):

- Open a [discussion](https://github.com/marcoalopez/PyRockWave/discussions): This is a place to:
  - Ask questions you are wondering about.
  - Requests for specific features or share ideas.
  - Interact with the developers (still just me).
- Open and [issue](https://github.com/marcoalopez/PyRockWave/issues): This is a place to report or track bugs.
- Create a [pull request](https://github.com/marcoalopez/PyRockWave/pulls): You modified, corrected or added a feature to one of the notebooks and send it for one of the developers to review it and add it to the main page.

## Funding

The seed of these codes has been made possible thanks to funding from the Government of the Principality of Asturias and the Foundation for the Promotion of Applied Research in Asturias (FICYT) (grant: SV-PA-21-AYUD/2021/57163) under the Asturias Plan for Science, Technology and Innovation (PCTI-Asturias) 2018-2022. I am currently seeking further funding to complete this project.

---
*Copyright © 2025 Marco A. Lopez-Sanchez*  

> [!WARNING]
>  _The information on this website and in the script documentation is provided without any warranty of any kind, either expressed or implied, and may include technical inaccuracies or typographical errors; the author reserves the right to make changes or improvements to the content of this website and the script documentation at any time without notice. This website and its documentation are not responsible for the content of external links. Notebook content is licensed under [Creative Commons Attribution license CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) and codes are licensed under GNU General Public License v3 (https://www.gnu.org/licenses/gpl-3.0.en.html) unless otherwise noted._

_Hosted on GitHub Pages — This website was created with [Typora](https://typora.io/)_