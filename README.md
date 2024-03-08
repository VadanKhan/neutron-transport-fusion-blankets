# Neutron Transport Simulation

This script simulates the motion of a neutron through different materials using the Woodcock method. It also generates various plots to visualize the results.

## Features

- Calculation of absorption probability and mean free path for different materials (water and lithium).
- Generation of histograms for absorption mean free path and total mean free path.
- Simulation of neutron processes (reflection, absorption, transmission).
- Simulation of neutron trajectory in 3D.
- Generation of pie charts for reflection, absorption, and transmission processes.

## Functions

- `water_props()`: Calculates the absorption probability and mean free path for water.
- `lithium_props()`: Calculates the absorption probability and mean free path for lithium.
- `generate_histogram()`: Generates a histogram for a given set of data.
- `calculate_neutron_processes()`: Simulates the reflection, absorption, and transmission of neutrons.
- `simulate_neutron_trajectory()`: Simulates the trajectory of a neutron and generates a 3D plot.
- `plot_pie_charts()`: Generates pie charts for the reflection, absorption, and transmission processes.
- `main()`: Main function that calls all other functions and generates the plots.

## Runtime

The script uses the `time` module to measure the runtime of each function. The runtime is printed to the console.

## Requirements

- Python 3.12
- NumPy 1.26.4
- SciPy 1.12.0
- Pandas 2.2.1
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
poetry run python main.py
```

Or if you have created the .venv in your cwd you can switch your interpreter to that directory and
so run however you wish.

This will generate the histograms, 3D plot, and pie charts.

