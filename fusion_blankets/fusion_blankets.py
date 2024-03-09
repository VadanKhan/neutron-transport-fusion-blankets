import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import time
import pandas as pd

# %% Neutron Transport Class & Functions


class Neutron:
    def __init__(self, speed):
        self.path = [(0, 0, 0)]  # Initialize with position at origin
        self.status = 0  # Initialize as in range of simulation
        self.direction = {'theta': np.pi / 2, 'phi': 0}  # Initialize moving in x direction
        self.speed = speed  # Mean free paths per second

    def scatter(self):
        # Randomize theta and phi when scattered
        self.direction['theta'] = np.arccos(1 - 2 * np.random.uniform())
        self.direction['phi'] = 2 * np.pi * np.random.uniform()

    def move(self, record_lambda):
        # Calculate new position based on current direction and speed
        x = self.path[-1][0] + self.speed * np.sin(self.direction['theta']) * np.cos(
            self.direction['phi']) * (-record_lambda * np.log(np.random.uniform()))
        y = self.path[-1][1] + self.speed * np.sin(self.direction['theta']) * np.sin(
            self.direction['phi']) * (-record_lambda * np.log(np.random.uniform()))
        z = self.path[-1][2] + self.speed * \
            np.cos(self.direction['theta']) * (-record_lambda * np.log(np.random.uniform()))
        self.path.append((x, y, z))

    def absorb(self):
        # Set status to absorbed
        self.status = 3

    def transmit(self):
        # Set status to transmitted
        self.status = 1

    def reflect(self):
        # Set status to reflected
        self.status = 2

    def in_blanket(self):
        # Set status to in blanket
        self.status = 4


def simulate_neutron_flux_store(cross_section, number_density, scattering_cross_section,
                                num_iterations=1000, neutron_number=10, breeder_lims=(100, 200),
                                finite_space_lims=(0, 300), y_lims=(-50, 50), z_lims=(-50, 50), velocity=1):
    '''
    This function simulates a flux of neutrons from one direction for a certain number of iterations.

    Parameters:
    cross_section: Nuclear cross section of the material.
    number_density: Number density of the material.
    scattering_cross_section: Scattering cross section of the material.
    num_iterations: The number of iterations for which the simulation should run.
    neutron_number: The number of neutrons to start with (default is 10).
    breeder_lims: The x limits of the breeder region (default is (100, 200)).
    finite_space_lims: The x limits of the simulation space (default is (0, 300)).
    y_lims: The y limits of the region (default is (-50, 50)).
    z_lims: The z limits of the region (default is (-50, 50)).
    velocity: The velocity of the neutrons in the x direction (default is 1).

    Returns:
    number_absorbed, number_transmitted, number_reflected, number_in_blanket, paths: The number of neutrons absorbed,
    transmitted, reflected, in blanket, and the paths of all neutrons.
    '''
    # Calculate the mean free path and absorption probability
    record_lambda = 1 / (number_density * cross_section)
    prob_a = cross_section * number_density
    prob_s = scattering_cross_section * number_density

    # Initialize the neutrons
    neutrons = [Neutron(velocity) for _ in range(neutron_number)]

    num_transmitted = 0
    num_absorbed = 0
    num_reflected = 0
    num_in_blanket = 0

    for _ in range(num_iterations):
        for neutron in neutrons:
            if neutron.status == 0:  # Only move neutrons that are still in the loop
                neutron.move(record_lambda)
                x, y, z = neutron.path[-1]
                if ((x > breeder_lims[0]) and (x < breeder_lims[1]) and
                    (y > y_lims[0]) and (y < y_lims[1]) and
                        (z > z_lims[0]) and (z < z_lims[1])):
                    if np.random.uniform() < prob_a:
                        neutron.absorb()
                        num_absorbed += 1
                    elif np.random.uniform() < prob_s:
                        neutron.scatter()
                if ((x > breeder_lims[1])  # if beyond blanket: "Transmitted"
                    and ((y < y_lims[0]) or (y > y_lims[1]) or  # Out of fiducial range conditions
                         (z < z_lims[0]) or (z > z_lims[1]) or (x > finite_space_lims[1]))):
                    neutron.transmit()
                    num_transmitted += 1
                elif ((x < breeder_lims[0])  # if before blanket: "Reflected"
                      and ((y < y_lims[0]) or (y > y_lims[1]) or  # Out of fiducial range conditions
                           (z < z_lims[0]) or (z > z_lims[1]) or (x < finite_space_lims[0]))):
                    neutron.reflect()
                    num_reflected += 1
                elif ((x > breeder_lims[0]) and (x < breeder_lims[1])  # if still "in blanket"
                      and ((y < y_lims[0]) or (y > y_lims[1]) or  # Out of fiducial range conditions
                           (z < z_lims[0]) or (z > z_lims[1]))):
                    neutron.in_blanket()
                    num_in_blanket += 1

    # Extract the paths from the neutrons
    paths = [neutron.path for neutron in neutrons]

    return num_absorbed, num_transmitted, num_reflected, num_in_blanket, paths

# %% Plotting Functions


def plot_neutron_paths(paths, x_lims=(0, 300), y_lims=(-100, 100), z_lims=(-100, 100),
                       breeder_lims=(100, 200), n=1):
    '''
    This function plots the path of each neutron.

    Parameters:
    paths: A list of paths of each neutron. Each path is a list of (x, y, z) coordinates.
    x_lims: The x limits of the region with finite mean free path (default is (5, 15)).
    y_lims: The y limits of the region (default is (-100, 100)).
    z_lims: The z limits of the region (default is (-100, 100)).
    n: The function will plot every nth point to reduce load (default is 1).
    '''

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.jet(np.linspace(0, 1, len(paths)))  # Create a color map

    # Add a shaded region to indicate the finite mean free path region
    for z in np.linspace(z_lims[0], z_lims[1], 100):  # Adjust the range as needed
        ax.add_collection3d(plt.fill_between(np.linspace(breeder_lims[0], breeder_lims[1], 10), y_lims[0],
                            y_lims[1], color='grey', alpha=0.01), zs=z, zdir='z')

    for i, path in enumerate(paths):
        if i % n == 0:  # Plot every nth path
            x, y, z = zip(*path)  # Unzip the coordinates
            ax.plot(x, y, z, color=colors[i])  # Use line plot and color-code each path

    # Set the limits of the x, y, and z axes
    ax.set_xlim(x_lims)  # Set x limits
    ax.set_ylim(y_lims)  # Set y limits
    ax.set_zlim(z_lims)  # Set z limits

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_pie_charts(num_reflected, num_absorbed, num_transmitted, num_in_blanket):
    """
    This function generates a pie chart for the reflection, absorption, and transmission processes
    for a single material.

    Parameters:
    num_reflected: The number of neutrons reflected.
    num_absorbed: The number of neutrons absorbed.
    num_transmitted: The number of neutrons transmitted.

    Returns:
    None. The function generates a pie chart.
    """
    import matplotlib.pyplot as plt

    processes = 'Reflection', 'Absorption', 'Transmission', 'Scattered within Blanket'
    explode = (0.05, 0.05, 0.05, 0.05)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie((num_reflected, num_absorbed, num_transmitted, num_in_blanket),
           labels=processes, explode=explode, autopct='%1.0f%%')

# %% Main


def main():
    # Define your parameters
    cross_section_absorption = 0.1  # Nuclear absorptioncross section of the material
    cross_section_scattering = 0.1  # Nuclear cross section of the material
    number_density = 0.2  # Number density of the material
    number_iterations = 100  # The time for which the simulation should run
    neutron_number_set = 1000
    breeder_lims_set = (100, 200)
    xlims_set = (-50, breeder_lims_set[1] + 100)
    ylims_set = (-50, 50)
    zlims_set = (-50, 50)

    # Call the simulate_neutron_flux function
    num_absorbed, num_transmitted, num_reflected, num_in_blanket, paths = \
        simulate_neutron_flux_store(cross_section_absorption, number_density, cross_section_scattering,
                                    num_iterations=number_iterations,
                                    neutron_number=neutron_number_set, breeder_lims=breeder_lims_set,
                                    finite_space_lims=xlims_set, y_lims=ylims_set, z_lims=zlims_set,
                                    velocity=1)

    print(f"Number of Neutrons Absorbed: {num_absorbed}")
    print(f"Number of Neutrons Transmitted: {num_transmitted}")
    print(f"Number of Neutrons Reflected: {num_reflected}")
    print(f"Number of Neutrons in Blanket: {num_in_blanket}")

    # Call the plot_pie_charts function
    plot_pie_charts(num_reflected, num_absorbed, num_transmitted, num_in_blanket)

    # Call the plot_neutron_paths function
    plot_neutron_paths(paths, x_lims=xlims_set, y_lims=ylims_set,
                       z_lims=zlims_set, breeder_lims=breeder_lims_set, n=1)

    plt.show()


if __name__ == "__main__":
    main()
