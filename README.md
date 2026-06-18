![header](https://raw.githubusercontent.com/marcoalopez/PyRockWave/main/img/header.jpg)

This project is maintained by [Marco A. Lopez-Sanchez](https://marcoalopez.github.io/) - Last update: 2026-06-18

## **What is PyRockWave?**

**PyRockWave** is a free, open-source Python tool that reads single-crystal elasticity databases and calculates the elastic properties of Earth materials. Its submodules compute isotropic seismic properties, analytical anisotropy models, volume- and CPO-weighted averages, phase and group seismic properties from the Christoffel equation, reflectivity, effective stiffness/compliance tensors for layered media (Schoenberg & Muir approach), and ultrasonic signal processing. Further tools visualize seismic properties, convert between spherical and Cartesian coordinates, generate S2 grids, and simplify working with tensors.

PyRockWave is modular: it exposes plain Python functions so you can develop with it and integrate it into other tools. Worked examples in Jupyter notebooks keep workflows easy to follow and reproducible.

View repository on GitHub: https://github.com/marcoalopez/PyRockWave

> [!WARNING]
> PyRockWave is still in the beta stage. Its functionality may change and break backward compatibility. Parts of the code are not well tested yet, and there is no installation guide or detailed documentation. However, the code will always be free and open source. You are welcome to explore and use it at your own risk.

## **Requirements & Python installation**

Available soon 😊

## **Documentation**

Available soon 😊

## **Jupyter notebook Examples**

- [Interacting with the mineral elastic data](https://github.com/marcoalopez/PyRockWave/blob/main/notebooks/example_database.ipynb)

- [The ElasticProps class explained](https://github.com/marcoalopez/PyRockWave/blob/main/notebooks/ElasticTensor_explained.ipynb)

- [Demonstration of the functionality of the ``coordinates`` module](https://github.com/marcoalopez/PyRockWave/blob/main/notebooks/example_coordinates.ipynb)

- [Demonstration of the functionality of the ``tensor_tools`` module](https://github.com/marcoalopez/PyRockWave/blob/main/notebooks/example_tensor_tools.ipynb)

- [Demonstration of the functionality of the ``anisotropic_models`` module](https://github.com/marcoalopez/PyRockWave/blob/main/notebooks/example_anisotropic_models.ipynb)

- Demonstration of the functionality of the ``christoffel`` module (I'm working on it! 😊)

- Demonstration of the functionality of the ``layered_media`` module (I'm working on it! 😊)

## **How to contribute to this project?**

The GitHub website hosting the project provides several options (you will need a GitHub account, it’s free!):

- Open a [discussion](https://github.com/marcoalopez/PyRockWave/discussions): This is a place to:
  - Ask questions.
  - Request features or share ideas.
  - Interact with the developers (still just me).
- Open an [issue](https://github.com/marcoalopez/PyRockWave/issues): This is a place to report and track bugs.
- Create a [pull request](https://github.com/marcoalopez/PyRockWave/pulls): Submit a fix or new feature for a notebook, and a developer will review it and merge it.

## **Funding**

Funding from the Government of the Principality of Asturias and the Foundation for the Promotion of Applied Research in Asturias (FICYT) (grant: SV-PA-21-AYUD/2021/57163) under the Asturias Plan for Science, Technology and Innovation (PCTI-Asturias) 2018-2022 seeded this project. I am currently seeking further funding to complete it.

---
*Copyright © 2023-present Marco A. Lopez-Sanchez*  

> [!NOTE]
> _The information on this website and in the script documentation is provided without any warranty of any kind, either expressed or implied, and may include technical inaccuracies or typographical errors. The author reserves the right to make changes or improvements to the content of this website and the script documentation at any time without notice. This website and its documentation are not responsible for the content of external links. Notebook content is licensed under [Creative Commons Attribution license CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) and code is licensed under GNU General Public License v3 (https://www.gnu.org/licenses/gpl-3.0.en.html) unless otherwise noted._

_Hosted on GitHub Pages — This website was created with [Typora](https://typora.io/)_