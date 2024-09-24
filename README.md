# Time Domain C-CST

<div align="center">
<a href="https://gitmoji.dev">
  <img
    src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square"
    alt="Gitmoji"
  />
</a>
</div>

## Description

Codes related to the paper presenting a time-marching scheme for Continuum Mechanics Corrected Couple-Stress Theory

## Folder structure (and how to explore)

This Python project is divided into different folders to keep everything as tidy and reusable as possible.

- The `scripts` folder contains scripts that run the simulations done.
  - The `implicit_time_scheme_paper` contains the results for the corresponding paper
  - `other`contains intermediate simulations done as validation, or experiments not presented in other reports
- The `time_domain_ccst` is a library that contains several useful functions that are used in the scripts. It contains functions for the **Method of Manufactured Solution**, as well as the solver for dynamic, static and eigenproblem cases and other utilities for data handling and plotting.
- The `notebooks` folder contains some independent notebooks where I tried some things dunring the development of the repo. Those are non-essential, but may be interesting to some.

You should start exploring the reports existing that are supported by this library. And then check the scripts where those results are calculated.

## Installation

This repository uses [poetry](https://python-poetry.org/) to manage the dependencies, and to install the local library `time_domain_ccst`.

### Option 1: Using poetry

1. **Install Project Dependencies:**

```bash
poetry install
```

2. **Activate the Virtual Environment:**

```bash
poetry shell
```

### Option 2: Using Conda

1. **Create and activate a Conda Environment**: Replace `my_env` with your desired environment name.

```bash
conda create -n my_ev python=3.12
conda activate my_env
```

2. **Install poetry**:

```bash
conda install -c conda-forge poetry
```

3. **Install Project Dependencies**:

```bash
poetry install
```

### Option 3: The rawest without prior requirements

Just try using python 3.12, although I don't see why it shouldn't work with older version (haven't tried):

1. **Create virtual environment**: Using `venv`, the python native tool.
```bash
python3 -m venv .venv
```

2. **Activate the virtual environment**: So your terminal session installs everythhing there.
```bash
source .venv/bin/activate
``` 

3. **Install the dependencies**: This uses the `pyproject.toml` file to install everything, including the local library. That one gets installed in editable mode, so if you change anything it will get updated automatically.
```bash
pip install -e .
```

4. **Run the scripts you want**: You can do it directly on terminal like this, or using your IDE.
```bash
python3 scripts/...{whatever you want to run}
```
