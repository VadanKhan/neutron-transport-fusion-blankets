import numpy as np
import matplotlib.pyplot as plt
import time


def calculate_neutron_processes(cross_section, number_density, prob_a, thickness, neutron_number):
    '''
    This function calculates the rate of absorption, reflection and transmission through a certain
    thickness of a material.

    Parameters:
    cross_section: Cross section of the material.
    number_density: Number density of the material.
    prob_a: Probability of absorption.
    thickness: Thickness of the material.
    neutron_number: Number of neutrons.

    Returns:
    number_reflected, number_absorbed, number_transmitted: The number of neutrons reflected,
    absorbed, and transmitted.
    '''
    start_time = time.time()
    record_lambda = 1 / (number_density * cross_section)
    x_pos = (-record_lambda*np.log(np.random.uniform(size=neutron_number)))
    num_transmitted = 0
    num_reflected = 0
    num_absorbed = 0
    while len(x_pos) > 0:
        transmit = np.where((x_pos > thickness))[0]
        x_pos = np.delete(x_pos, transmit, 0)

        reflect = np.where((x_pos < 0))[0]
        x_pos = np.delete(x_pos, reflect, 0)

        absorb = np.where((np.random.uniform(size=len(x_pos)) < prob_a))[0]
        x_pos = np.delete(x_pos, absorb, 0)

        a = 1 - np.random.uniform(size=len(x_pos)) * 2
        theta = np.arccos(a)
        phi = np.random.uniform(size=len(x_pos)) * 2*np.pi
        x_pos += (np.sin(theta)*np.cos(phi) *
                  (-record_lambda*np.log(np.random.uniform(size=len(x_pos)))))

        num_transmitted += len(transmit)
        num_reflected += len(reflect)
        num_absorbed += len(absorb)

    print(f"Execution time for calculate_neutron_processes: {time.time() - start_time} seconds")
    return num_reflected, num_absorbed, num_transmitted


def simulate_neutron_flux(cross_section, number_density, num_iterations, neutron_number=10):
    '''
    This function simulates a flux of neutrons from one direction for a certain number of iterations.

    Parameters:
    cross_section: Nuclear cross section of the material.
    number_density: Number density of the material.
    num_iterations: The number of iterations for which the simulation should run.
    neutron_number: The number of neutrons to start with (default is 10).

    Returns:
    number_reflected, number_absorbed, number_transmitted: The number of neutrons reflected,
    absorbed, and transmitted.
    '''
    # Calculate the mean free path and absorption probability
    record_lambda = 1 / (number_density * cross_section)
    prob_a = cross_section * number_density

    # Initialize the positions and directions of the neutrons
    x_pos = np.zeros(neutron_number)
    y_pos = np.zeros(neutron_number)
    z_pos = np.zeros(neutron_number)
    random_theta = np.full(neutron_number, np.pi/2)
    random_phi = np.full(neutron_number, 0.0)

    num_transmitted = 0
    num_reflected = 0
    num_absorbed = 0

    for _ in range(num_iterations):
        # Calculate the new positions of the neutrons
        x_pos += (np.sin(random_theta) * np.cos(random_phi) *
                  (-record_lambda*np.log(np.random.uniform(size=len(x_pos)))))
        y_pos += (np.sin(random_theta)*np.sin(random_phi) *
                  (-record_lambda*np.log(np.random.uniform(size=len(x_pos)))))
        z_pos += (np.cos(random_theta) *
                  (-record_lambda*np.log(np.random.uniform(size=len(x_pos)))))
        # Determine which neutrons are absorbed
        absorb = np.where((np.random.uniform(size=len(x_pos)) < prob_a))[0]
        x_pos = np.delete(x_pos, absorb)
        y_pos = np.delete(y_pos, absorb)
        z_pos = np.delete(z_pos, absorb)
        random_theta = np.delete(random_theta, absorb)
        random_phi = np.delete(random_phi, absorb)
        num_absorbed += len(absorb)
        # Update the directions of the remaining neutrons
        a = 1 - np.random.uniform(size=len(x_pos)) * 2
        random_theta = np.arccos(a)
        random_phi = np.random.uniform(size=len(x_pos)) * 2*np.pi

    num_transmitted = len(x_pos)
    return num_reflected, num_absorbed, num_transmitted, x_pos, y_pos, z_pos


def plot_pie_charts(num_reflected, num_absorbed, num_transmitted):
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

    processes = 'Reflection', 'Absorption', 'Transmission'
    explode = (0.05, 0.05, 0.05)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie((num_reflected, num_absorbed, num_transmitted),
           labels=processes, explode=explode, autopct='%1.0f%%')


def main():
    # Define your parameters
    cross_section = 0.1  # Nuclear cross section of the material
    number_density = 0.2  # Number density of the material
    number_iterations = 1000  # The time for which the simulation should run

    # Call the simulate_neutron_flux function
    num_reflected, num_absorbed, num_transmitted, x_pos, y_pos, z_pos = simulate_neutron_flux(
        cross_section, number_density, 69, 1000)

    # Call the plot_pie_charts function
    plot_pie_charts(num_reflected, num_absorbed, num_transmitted)

    # Print the positions of the neutrons for plotting
    print("x positions:", x_pos)
    print("y positions:", y_pos)
    print("z positions:", z_pos)
    plt.show()


if __name__ == "__main__":
    main()
