# Time Domain C-CST
Codes related to the paper presenting a time-marching scheme for Continuum Mechanics Corrected Couple-Stress Theory

## How to run things

This repository uses [poetry](https://python-poetry.org/) to manage the dependencies, and to install the local library `time_domain_ccst`.

### Installation with poetry

```bash
poetry install
```

### Have never used poetry, but use conda

Create a venv, install poetry there and use it to install the remaining dependencies.

```bash
conda create -n td_ccst_venv python=3.12

pip install poetry
poetry install
```
