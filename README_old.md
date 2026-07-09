# MGGP (Multigene Genetic Programming)

[![Version](https://img.shields.io/badge/version-0.0.3-blue.svg)](https://github.com/RafaelGAT108/mggp_model/releases/tag/v0.0.3)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![codecov](https://codecov.io/gh/RafaelGAT108/mggp_model/branch/main/graph/badge.svg)](https://codecov.io/gh/RafaelGAT108/mggp_model)

A flexible and extensible implementation of **Multigene Genetic Programming (MGGP)** focused on **system identification** and **symbolic modeling**, supporting:

- **Regression (static and dynamic systems)**
- **Classification tasks**
- **Dynamic models in NARX and FIR formulations**
- **SISO, MISO, and MIMO system configurations**

---

## 📦 Version: v0.0.3 (Latest Release)

This version introduces important improvements in **modeling capabilities**, **code clarity**, and **project structure**, moving the project closer to a fully packaged Python library.

### 🚀 What's new?

- ✅ **Improved readability of Free-Run and Multi-Shooting (MShooting) implementations**  
  - Clearer iteration flow  
  - Explicit batch handling in MShooting  
  - Consistent behavior across **SISO, MISO, and MIMO**

- ✅ **Support for current input term `u[k]`**  
  Previously, models only considered delayed inputs (`u[k-1], ..., u[k-lagMax]`).  
  Including `u[k]` significantly enhances the expressive power of the identified models.

- ✅ **Advancements toward packaging as a Python library**  
  Internal refactoring and structural improvements preparing for future PyPI release.

- ✅ **Improved symbolic simplification for SISO models**  
  Cleaner and more interpretable final equations.

- ✅ **General code cleanup and notebook reorganization**

---


## 🧠 Core Features

- Symbolic regression via **Multigene Genetic Programming**
- Support for **dynamic system identification (NARX / FIR)**
- Native handling of:
  - SISO (Single Input Single Output)
  - MISO (Multiple Input Single Output)
  - MIMO (Multiple Input Multiple Output)
- Free-run simulation and **multi-shooting training strategies**
- Interpretable model structures (explicit equations)

---

## 🚀 Quick Start

Before start to run any notebook, run the code below to install the mggp as a library:

```bash
pip install -e .
```

otherwise, you'll need use:
```python
src.mggp import MGGP
```

Check out the example notebook to get started:

📓 **`01_MGGP_Regression.ipynb`** - Demonstrates how to configure and use MGGP for Regression problems, and

📓 **`01_MGGP_Classifier.ipynb`** - Demonstrates how to configure and use MGGP for Classification problems.



## 📋 Requirements

The requirements are descript in the requirements.txt

This implementation is based on the MGGP methodology described in the following papers:

### Foundational Works

**Multi-Gene Genetic Programming for Nonlinear MIMO Modeling of F16 Aircraft Ground Vibrations**

> DOS SANTOS, Rafael Ávila et al. Multi-Gene Genetic Programming para Modelagem MIMO Nao Linear das Vibraçoes de uma Aeronave F16 no Solo.*

[📄 Download PDF](https://www.researchgate.net/profile/Rafael-Dos-Santos-26/publication/400395141_Multi-Gene_Genetic_Programming_para_Modelagem_MIMO_Nao_Linear_das_Vibracoes_de_uma_Aeronave_F16_no_Solo/links/69820b695d60ab48356a7d4f/Multi-Gene-Genetic-Programming-para-Modelagem-MIMO-Nao-Linear-das-Vibracoes-de-uma-Aeronave-F16-no-Solo.pdf)


**A Novel MIMO Multi-Gene Genetic Programming Approach for Interpretable NARX Models: an application to vehicle state estimation**

> HENRIQUE GROENNER BARBOSA, Bruno et al. A Novel MIMO Multi-Gene Genetic Programming Approach for Interpretable NARX Models: an application to vehicle state estimation. Henrique and Ávila Santos, Rafael and Correa Victorino, Alessandro and Askari, Hassan and Xu, Nan, A Novel MIMO Multi-Gene Genetic Programming Approach for Interpretable NARX Models: an application to vehicle state estimation.*

[📄 Download PDF](https://doi.org/10.2139/ssrn.5946034)

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
