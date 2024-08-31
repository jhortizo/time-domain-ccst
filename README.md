# Time Domain C-CST

## Description

Codes related to the paper presenting a time-marching scheme for Continuum Mechanics Corrected Couple-Stress Theory

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

