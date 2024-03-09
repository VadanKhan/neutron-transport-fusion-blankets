# Neutron Transport in Fusion Blankets

This repository contains scripts for simulating neutron transport in fusion blankets. The main script is `fusion_blankets.py`, which uses the Woodcock method to simulate neutron transport in a fusion blanket composed of different materials.

## Features

- Simulation of neutron transport in fusion blankets.
- Calculation of absorption probability and mean free path for different materials.
- Tracking of neutron processes (reflection, absorption, transmission, in blanket).
- Generation of pie charts for neutron processes.
- Visualization of neutron paths in 3D.

## Scripts

- `fusion_blankets.py`: The main script. It defines a `Neutron` class for simulating neutron transport, and functions for simulating neutron flux and plotting the results.
- `monte_carlo.py`: A script for Monte Carlo simulations (not detailed in this README).
- `raw_scripts.py`: A script containing raw scripts (not detailed in this README).

## Usage

1. Define your parameters at the beginning of the `main` function in `fusion_blankets.py`.
2. Run `fusion_blankets.py`. This will simulate neutron flux for the defined parameters and generate a pie chart and a 3D plot of neutron paths.
3. The pie chart shows the proportions of neutrons reflected, absorbed, transmitted, and in the blanket. The absorbed category is further broken down by material.
4. The 3D plot shows the paths of neutrons in the fusion blanket.

## Requirements

- Python 3.12
- NumPy 1.26.4
- Matplotlib 3.8.3

## Setup with Poetry

1. Install Poetry if you haven't already. You can do this using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository and navigate to the project directory.

3. Install the project dependencies using Poetry:

```bash
poetry install
```

This will create a virtual environment and install the dependencies specified in the `pyproject.toml` file.

**Note** to create the .venv in your current working directory (cwd), run the following command first:
```bash
poetry config virtualenvs.in-project true
```

4. You can then run the script within the virtual environment:

```bash
poetry run python fusion_blankets.py
```

