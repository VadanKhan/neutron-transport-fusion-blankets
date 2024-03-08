# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from numpy.random import randint
# from itertools import combinations
# import pandas as pd
import time

# Defining Avogadro's number

NA = 6.022*10**23
# Physical constants for water
M_WATER = 18.0153*10**-3  # Molar mass of water in kg/mol
RHO_WATER = 10**3  # Density of water in kg/m^3
SIGMA_A_WATER = 0.6652*10**-28  # Absorption cross-section of water in m^2
SIGMA_S_WATER = 103*10**-28  # Scattering cross-section of water in m^2

# Physical constants for lithium
M_LITHIUM = 6.0151*10**-3  # Molar mass of lithium in kg/mol
RHO_LITHIUM = 534  # Density of lithium in kg/m^3
SIGMA_A_LITHIUM = 938*10**-28  # Absorption cross-section of lithium in m^2
SIGMA_S_LITHIUM = 0.97*10**-28  # Scattering cross-section of lithium in m^2

# %% Material Properties Calculations


def water_props():
    """
    This function calculates the absorption probability and mean free path for water.
    It uses the molar mass, density, and cross-sections for absorption and scattering to calculate
    these values.
    """
    # Macroscopic absorption cross-section of water
    macro_sigma_a_water = RHO_WATER*NA*SIGMA_A_WATER/M_WATER
    # Macroscopic scattering cross-section of water
    macro_sigma_s_water = RHO_WATER*NA*SIGMA_S_WATER/M_WATER
    # Total macroscopic cross-section of water
    macro_sigma_total_water = macro_sigma_s_water + macro_sigma_a_water
    # Probability of absorption in water
    prob_a_water = macro_sigma_a_water / macro_sigma_total_water
    # Probability of scattering in water
    # prob_s_water = macro_sigma_s_water / macro_sigma_total_water
    # Mean free path in water in cm
    mfp_water = (1 / macro_sigma_total_water)*100
    return prob_a_water, mfp_water


def lithium_props():
    """
    This function calculates the absorption probability and mean free path for lithium.
    It uses the molar mass, density, and cross-sections for absorption and scattering to calculate
    these values.
    """
    # Macroscopic absorption cross-section of lithium
    macro_sigma_a_lithium = RHO_LITHIUM*NA*SIGMA_A_LITHIUM/M_LITHIUM
    # Macroscopic scattering cross-section of lithium
    macro_sigma_s_lithium = RHO_LITHIUM*NA*SIGMA_S_LITHIUM/M_LITHIUM
    # Total macroscopic cross-section of lithium
    macro_sigma_total_lithium = macro_sigma_s_lithium + macro_sigma_a_lithium
    # Probability of absorption in lithium
    prob_a_lithium = macro_sigma_a_lithium / macro_sigma_total_lithium
    # Probability of scattering in lithium
    # prob_s_lithium = macro_sigma_s_lithium / macro_sigma_total_lithium
    # Mean free path in lithium in cm
    mfp_lithium = (1 / macro_sigma_total_lithium)*100
    return prob_a_lithium, mfp_lithium


# %% Histogram Plotting Functions


def generate_histogram(l_nominal, r_max, title, N_iter, N_bins, N):
    """
    This function generates a histogram corresponding to the probability distribution of a particle.
    The histogram is based on input random numbers weighted by the inverse cumulative distribution
        function.
    The function also calculates the mean free path (MFP) and its error from the histogram data.
    The MFP (lambda_nominal) is the average distance that a particle travels before interacting.

    Parameters:
    l_nominal: Nominal lambda value (MFP) used for generating the histogram.
    r_max: Maximum range for the histogram.
    title: Title for the histogram plot.
    N_iter: Number of iterations for the histogram generation.
    N_bins: Number of bins in the histogram.
    N: Number of random numbers to generate for the histogram.

    Returns:
    record_lambda: Calculated lambda value (MFP) from the histogram data.
    lambda_error: Error in the calculated lambda value (MFP).
    """
    hist_data = np.zeros((N_iter, N_bins))
    mean_freq = np.zeros(N_bins)
    std_freq = np.zeros(N_bins)
    for i in range(N_iter):
        hist_data[i, :], r_bin = np.histogram(-l_nominal*np.log(np.random.uniform(size=N)),
                                              N_bins, range=(0, r_max))
    r_bin = r_bin[:-1]
    for j in range(0, N_bins):
        mean_freq[j] = np.mean(hist_data[:, j])
        std_freq[j] = np.std(hist_data[:, j])
    std_freq = np.delete(std_freq, np.argwhere(mean_freq == 0))
    r_bin = np.delete(r_bin, np.argwhere(mean_freq == 0))
    mean_freq = np.delete(mean_freq, np.argwhere(mean_freq == 0))
    weight = mean_freq/std_freq
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    plt.suptitle(title)
    ax[0].bar(r_bin, mean_freq, align='center', width=r_max/30, color='white',
              yerr=std_freq, edgecolor='black', linewidth=1.2)
    coef, covr = np.polyfit(r_bin, np.log(mean_freq), 1, cov=True, w=weight)
    Pfit = np.polyval(coef, r_bin)
    ax[1].errorbar(r_bin, Pfit, yerr=std_freq/mean_freq, fmt='.')
    ax[1].plot(r_bin, Pfit)
    record_lambda = -1/coef[0]
    lambda_error = np.sqrt(covr[0][0])
    ax[0].plot(r_bin, np.exp(coef[1])*np.exp(coef[0]*r_bin),
               label='MFP = ({0:.3} ± {1:.3})cm'.format(record_lambda, lambda_error))
    ax[0].legend()
    ax[0].set_ylabel('PDF')
    ax[0].set_xlabel('Scaled ICDF numbers')
    ax[1].set_ylabel('log(PDF)')
    ax[1].set_xlabel('Scaled ICDF numbers')
    return record_lambda, lambda_error

# %% Simple Stochastic Neutron Functions


def simulate_neutron_movement(record_lambda, prob_a, T, N):
    """
    This function simulates the movement of a single neutron in a medium.

    Parameters:
    record_lambda: Mean free path of the neutron.
    prob_a: Probability of absorption.
    T: Threshold distance for the neutron.
    N: Number of random positions to generate for the neutron.

    Returns:
    x, y, z: Lists containing the x, y, and z positions of the neutron.
    """
    x_pos = (-record_lambda*np.log(np.random.uniform(size=N)))
    y_pos = (-record_lambda*np.log(np.random.uniform(size=N)))
    z_pos = (-record_lambda*np.log(np.random.uniform(size=N)))
    x = []
    y = []
    z = []
    while len(x_pos) > 0:
        transmit = np.where((x_pos > T))[0]
        x_pos = np.delete(x_pos, transmit, 0)
        y_pos = np.delete(y_pos, transmit, 0)
        z_pos = np.delete(z_pos, transmit, 0)

        reflect = np.where((x_pos < 0))[0]
        x_pos = np.delete(x_pos, reflect, 0)
        y_pos = np.delete(y_pos, reflect, 0)
        z_pos = np.delete(z_pos, reflect, 0)

        absorb = np.where((np.random.uniform(size=len(x_pos)) < prob_a))[0]
        x_pos = np.delete(x_pos, absorb, 0)
        y_pos = np.delete(y_pos, absorb, 0)
        z_pos = np.delete(z_pos, absorb, 0)

        a = 1 - np.random.uniform(size=len(x_pos)) * 2
        theta = np.arccos(a)
        phi = np.random.uniform(size=len(x_pos)) * 2*np.pi
        x_pos += (np.sin(theta)*np.cos(phi) * (-record_lambda*np.log(np.random.uniform())))
        y_pos += (np.sin(theta)*np.sin(phi) * (-record_lambda*np.log(np.random.uniform())))
        z_pos += (np.cos(theta) * (-record_lambda*np.log(np.random.uniform())))

        x = np.concatenate((x, x_pos))
        y = np.concatenate((y, y_pos))
        z = np.concatenate((z, z_pos))
    return x, y, z


def calculate_neutron_processes(record_lambda, prob_a, thickness, neutron_number):
    '''
    This function calculates the rate of absorption, reflection and transmission through a certain
    thickness of a material.

    Parameters:
    record_lambda: Mean free path of the neutrons.
    prob_a: Probability of absorption.
    thickness: Thickness of the material.
    neutron_number: Number of neutrons.

    Returns:
    number_reflected, number_absorbed, number_transmitted: The number of neutrons reflected,
    absorbed, and transmitted.
    '''
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
    return num_reflected, num_absorbed, num_transmitted


def plot_pie_charts(lambda_values, prob_values, N, T, materials):
    """
    This function generates pie charts for the reflection, absorption, and transmission processes
    for different materials.

    Parameters:
    lambda_values: List of mean free paths for the neutrons.
    prob_values: List of probabilities of absorption.
    N: Number of neutrons.
    T: Array of thickness values to evaluate.
    materials: List of materials.

    Returns:
    None. The function generates pie charts.
    """
    transmit = np.zeros((10, 2))
    absorb = np.zeros((10, 2))
    reflect = np.zeros((10, 2))
    for i in range(10):
        for j in range(len(lambda_values)):
            reflect[i][j], absorb[i][j], transmit[i][j] = calculate_neutron_processes(
                lambda_values[j], prob_values[j], T, N)

    mean_reflect = reflect.mean(axis=0)
    mean_absorb = absorb.mean(axis=0)
    mean_transmit = transmit.mean(axis=0)
    std_reflect = reflect.std(axis=0)
    std_absorb = absorb.std(axis=0)
    std_transmit = transmit.std(axis=0)

    processes = 'Reflection', 'Absorption', 'Transmission'
    explode = (0.05, 0.05, 0.05)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('10cm of each material')
    fig.tight_layout(pad=5)
    for i in range(len(lambda_values)):
        ax[i].set_title(materials[i])
        ax[i].pie((mean_reflect[i], mean_absorb[i], mean_transmit[i]),
                  labels=processes, explode=explode, autopct='%1.0f%%')
        fig, axs = plt.subplots(figsize=(7, 1))
        col_labels = [materials[i]]
        row_labels = ['Uncertainty in Reflection%',
                      'Uncertainty in Absorption%', 'Uncertainty in Transmission%']
        table_vals = [['±{0:.3f}'.format(std_reflect[i]*100/N)], ['±{0:.3f}'.format(std_absorb[i] *
                                                                                    100/N)], [
            '±{0:.3f}'.format(std_transmit[i]*100/N)]]
        axs.axis('off')
        axs.axis('tight')
        axs.table(cellText=table_vals, rowLabels=row_labels, colLabels=col_labels, loc='center')


# %% Thickness Sweep Functions


def evaluate_processes(lambda_val, prob, thickness):
    '''
    This function evaluates how the rates of reflection, absorption, and transmission change
    as a function of the thickness of a material.

    NB: NEED "calculate_neutron_processes" DEFINED.

    Parameters:
    lambda_val: Mean free path of the neutrons.
    prob: Probability of absorption.
    thickness: Array of thickness values to evaluate.

    Returns:
    mean_reflect, mean_absorb, mean_transmit: Mean rates of reflection, absorption, and
        transmission.
    std_reflect, std_absorb, std_transmit: Standard deviations of the rates of reflection,
    absorption, and transmission.
    '''
    num_thickness = len(thickness)
    transmit_rates = np.zeros((num_thickness, 10))
    absorb_rates = np.zeros((num_thickness, 10))
    reflect_rates = np.zeros((num_thickness, 10))
    for j in range(10):
        for i in range(num_thickness):
            reflect_rates[i][j], absorb_rates[i][j], transmit_rates[i][j] = \
                calculate_neutron_processes(lambda_val, prob, thickness[i])
    mean_reflect = reflect_rates.mean(axis=1)
    mean_absorb = absorb_rates.mean(axis=1)
    mean_transmit = transmit_rates.mean(axis=1)
    std_reflect = reflect_rates.std(axis=1)
    std_absorb = absorb_rates.std(axis=1)
    std_transmit = transmit_rates.std(axis=1)
    return mean_reflect, mean_absorb, mean_transmit, std_reflect, std_absorb, std_transmit


def plot_rates(lambda_vals, prob_vals, thickness, N, materials):
    """
    This function plots the rates of reflection, absorption, and transmission as a function of the
    thickness of a material.

    Parameters:
    lambda_vals: List of mean free paths for the neutrons.
    prob_vals: List of probabilities of absorption.
    thickness: Array of thickness values to evaluate.
    N: Number of neutrons.
    materials: List of materials.

    Returns:
    None. The function generates a plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i in range(len(lambda_vals)):
        ax[i].set_title(materials[i])
        reflect, absorb, transmit, std_r, std_a, std_t = evaluate_processes(lambda_vals[i],
                                                                            prob_vals[i], thickness)
        ax[i].errorbar(thickness, reflect*100/N, yerr=std_r*100/N, label='reflect')
        ax[i].errorbar(thickness, absorb*100/N, yerr=std_a*100/N, label='absorb')
        ax[i].errorbar(thickness, transmit*100/N, yerr=std_t*100/N, label='transmit')
        ax[1].set_xlabel('Thickness/cm')
        ax[0].set_ylabel('Rate of Processes/%')
        ax[i].legend()
    return

# %% Neutron Trajectory


def simulate_neutron_trajectory(record_lambda, prob, T1, T2, N, sigmaa, sigmaT, splitting_input=1):
    """
    This function simulates the motion of a neutron through a vacuum and water using the Woodcock
      method.
    It also plots the 3D trajectory of the neutron.

    Parameters:
    record_lambda: Mean free path of the neutron.
    prob: Probability of absorption.
    T1: Thickness of the vacuum.
    T2: Thickness of the water.
    N: Number of random positions to generate for the neutron.
    sigmaa: Absorption cross-section.
    sigmaT: Total cross-section.
    splitting_input: if this = "N", then the function will only plot every "Nth" value to reduce
        runtime

    Returns:
    None. The function generates a plot.
    """
    x_pos = (-record_lambda*np.log(np.random.uniform(size=N)))
    y_pos = np.zeros(N)
    z_pos = np.zeros(N)
    random_theta = np.full(N, np.pi/2)
    random_phi = np.zeros(N)
    x = []
    y = []
    z = []
    while len(x_pos) > 0:
        transmit = np.where((x_pos >= T1 + T2))[0]
        x_pos = np.delete(x_pos, transmit, 0)
        y_pos = np.delete(y_pos, transmit, 0)
        z_pos = np.delete(z_pos, transmit, 0)
        random_theta = np.delete(random_theta, transmit, 0)
        random_phi = np.delete(random_phi, transmit, 0)

        reflect = np.where((x_pos <= 0))[0]
        x_pos = np.delete(x_pos, reflect, 0)
        y_pos = np.delete(y_pos, reflect, 0)
        z_pos = np.delete(z_pos, reflect, 0)
        random_theta = np.delete(random_theta, reflect, 0)
        random_phi = np.delete(random_phi, reflect, 0)

        v = np.random.uniform(size=len(x_pos))
        region2 = np.where((x_pos > T1) & (x_pos <= T1 + T2))
        fic = np.where((v > sigmaa/sigmaT) & (x_pos >= 0) & (x_pos <= T1))
        n_fic = np.where((v <= sigmaa/sigmaT) & (x_pos >= 0) & (x_pos <= T1))

        w = np.random.uniform(size=len(x_pos[fic]))
        x_pos[fic] += (np.sin(random_theta[fic])*np.cos(random_phi[fic])
                       * (-record_lambda*np.log(w)))
        y_pos[fic] += (np.sin(random_theta[fic])*np.sin(random_phi[fic])
                       * (-record_lambda*np.log(w)))
        z_pos[fic] += (np.cos(random_theta[fic]) * (-record_lambda*np.log(w)))

        a = 1 - np.random.uniform(size=len(x_pos[n_fic])) * 2
        random_theta[n_fic] = np.arccos(a)
        random_phi[n_fic] = np.random.uniform(size=len(x_pos[n_fic])) * 2*np.pi
        x_pos[n_fic] += (np.sin(random_theta[n_fic])*np.cos(random_phi[n_fic]) *
                         (-record_lambda*np.log(np.random.uniform(size=len(x_pos[n_fic])))))
        y_pos[n_fic] += (np.sin(random_theta[n_fic])*np.sin(random_phi[n_fic]) *
                         (-record_lambda*np.log(np.random.uniform(size=len(x_pos[n_fic])))))
        z_pos[n_fic] += (np.cos(random_theta[n_fic]) *
                         (-record_lambda*np.log(np.random.uniform(size=len(x_pos[n_fic])))))

        a = 1 - np.random.uniform(size=len(x_pos[region2])) * 2
        random_theta[region2] = np.arccos(a)
        random_phi[region2] = np.random.uniform(size=len(x_pos[region2])) * 2*np.pi
        x_pos[region2] += (np.sin(random_theta[region2])*np.cos(random_phi[region2])
                           * (-record_lambda*np.log(np.random.uniform(size=len(x_pos[region2])))))
        y_pos[region2] += (np.sin(random_theta[region2])*np.sin(random_phi[region2])
                           * (-record_lambda*np.log(np.random.uniform(size=len(x_pos[region2])))))
        z_pos[region2] += (np.sin(random_theta[region2]) *
                           (-record_lambda*np.log(np.random.uniform(size=len(x_pos[region2])))))

        absorb1 = np.where((np.random.uniform(size=len(x_pos)) < prob)
                           & (x_pos > T1) & (x_pos <= T1 + T2))[0]
        x_pos = np.delete(x_pos, absorb1, 0)
        y_pos = np.delete(y_pos, absorb1, 0)
        z_pos = np.delete(z_pos, absorb1, 0)
        random_theta = np.delete(random_theta, absorb1, 0)
        random_phi = np.delete(random_phi, absorb1, 0)

        x = np.concatenate((x, x_pos))
        y = np.concatenate((y, y_pos))
        z = np.concatenate((z, z_pos))

    n = splitting_input
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot3D(x[::n], y[::n], z[::n], '.')
    ax.plot3D(x[::n], y[::n], z[::n])
    ax.scatter3D(x[0], y[0], z[0], color='red', label='beginning')
    ax.scatter3D(x[-1], y[-1], z[-1], color='blue', label='end')
    ax.set_box_aspect(aspect=(1, 1, 1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()


# %% Main Script Calling


def main():
    # Get the absorption probability and mean free path for water and lithium
    start_time = time.time()
    prob_aw, mfp_water = water_props()
    prob_al, mfp_lithium = lithium_props()
    print("Time taken for water_props and lithium_props: %s seconds" % (time.time() - start_time))

    N = 10000
    N_iteration = 10
    N_bins = 30
    r_max = 300
    lambda_nominal = 45
    title = 'Absorption Mean Free Path - Water'
    start_time = time.time()
    record_lambda_a10, lambda_error10 = generate_histogram(
        lambda_nominal, r_max, title, N_iteration, N_bins, N)
    print("Time taken for generate_histogram (Absorption Mean Free Path - Water): %s seconds" %
          (time.time() - start_time))

    r_max = 2
    title = 'Total Mean Free Path - Water'
    start_time = time.time()
    record_lambda_w, lambda_error_w = generate_histogram(
        mfp_water, r_max, title, N_iteration, N_bins, N)
    print("Time taken for generate_histogram (Total Mean Free Path - Water): %s seconds" %
          (time.time() - start_time))

    r_max = 0.1
    title = 'Total Mean Free Path - Lithium'
    start_time = time.time()
    record_lambda_l, lambda_error_l = generate_histogram(
        mfp_lithium, r_max, title, N_iteration, N_bins, N)
    print("Time taken for generate_histogram (Total Mean Free Path - Lithium): %s seconds" %
          (time.time() - start_time))

    T = 10  # Thickness of material in cm
    start_time = time.time()
    num_reflected, num_absorbed, num_transmitted = calculate_neutron_processes(
        record_lambda_w, prob_aw, T, N)
    print("Time taken for calculate_neutron_processes: %s seconds" % (time.time() - start_time))

    record_lambda_1 = record_lambda_l
    record_lambda_2 = record_lambda_w
    sigmaa = 0
    sigma2 = 1/record_lambda_2
    sigma1 = 1/record_lambda_1
    sigmaT = max(sigma1, sigma2)
    T1 = 10
    T2 = 10
    start_time = time.time()
    simulate_neutron_trajectory(record_lambda_w, prob_aw, T1, T2, int(N/10), sigmaa, sigmaT, 10)
    print("Time taken for simulate_neutron_trajectory: %s seconds" % (time.time() - start_time))

    # Generate pie charts
    start_time = time.time()
    lambda_values = [record_lambda_w, record_lambda_l]
    prob_values = [prob_aw, prob_al]
    materials = ['Water', 'Lithium-6']
    plot_pie_charts(lambda_values, prob_values, N, T, materials)
    print("Time taken for plot_pie_charts: %s seconds" % (time.time() - start_time))

    plt.show()


if __name__ == "__main__":
    main()
