from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.random import randint
from itertools import combinations
import pandas as pd


NA = 6.022*10**23
def Water():
    M = 18.0153*10**-3
    p = 10**3
    sigma_a = 0.6652*10**-28
    sigma_s = 103*10**-28
    macroscopic_sigma_a = p*NA*sigma_a/M
    macroscopic_sigma_s = p*NA*sigma_s/M
    macroscopic_total = macroscopic_sigma_s + macroscopic_sigma_a
    prob_a = macroscopic_sigma_a/ macroscopic_total
    prob_s = macroscopic_sigma_s/ macroscopic_total
    analytical_lambda = (1 / macroscopic_total)*100
    return prob_a, analytical_lambda

def Lithium():
    M = 6.0151*10**-3
    p = 534
    sigma_a = 938*10**-28
    sigma_s = 0.97*10**-28
    macroscopic_sigma_a = p*NA*sigma_a/M
    macroscopic_sigma_s = p*NA*sigma_s/M
    macroscopic_total = macroscopic_sigma_s + macroscopic_sigma_a
    prob_a = macroscopic_sigma_a/ macroscopic_total
    prob_s = macroscopic_sigma_s/ macroscopic_total
    analytical_lambda = (1 / macroscopic_total)*100
    return prob_a, analytical_lambda

prob_aw = Water()[0]
prob_al = Lithium()[0]
'N = 10000
N_iteration = 10
N_bins = 30'
def Histogram(lambda_nominal, r_max, title):
    '''
    This function generates a histgram corresponding to the probability distribution of a particle, depending on the
    input random number weighted by the inverse cumulative distribution function.
    '''
    NR = np.zeros((N_iteration, N_bins))
    mean_fre = np.zeros(N_bins)
    std_fre = np.zeros(N_bins)
    for i in range(N_iteration):
        NR[i,:],r_bin = np.histogram(-lambda_nominal*np.log(np.random.uniform(size = N)), N_bins, range = (0, r_max))
    r_bin = r_bin[:-1]
    for j in range(0, N_bins):
        mean_fre[j] = np.mean(NR[:,j])
        std_fre[j] = np.std(NR[:,j])
    std_fre=np.delete(std_fre,np.argwhere(mean_fre==0))
    r_bin=np.delete(r_bin,np.argwhere(mean_fre==0))
    mean_fre=np.delete(mean_fre,np.argwhere(mean_fre==0))
    weight = mean_fre/std_fre
    fig, ax = plt.subplots(1,2,figsize = (13,5))
    plt.suptitle(title)
    ax[0].bar(r_bin, mean_fre,align='center',width = r_max/30, color='white', yerr = std_fre, edgecolor='black', linewidth=1.2)
    (coef, covr) = np.polyfit(r_bin, np.log(mean_fre), 1, cov=True, w = weight)
    Pfit = np.polyval(coef, r_bin)
    ax[1].errorbar(r_bin, Pfit, yerr = std_fre/mean_fre, fmt = '.')
    ax[1].plot(r_bin, Pfit)
    record_lambda = -1/coef[0]
    lambda_error = np.sqrt(covr[0][0])
    ax[0].plot(r_bin, np.exp(coef[1])*np.exp(coef[0]*r_bin), label = 'MFP = ({0:.3} ± {1:.3})cm'.format(record_lambda, lambda_error))
    ax[0].legend()
    ax[0].set_ylabel('PDF')
    ax[0].set_xlabel('Scaled ICDF numbers')
    ax[1].set_ylabel('log(PDF)')
    ax[1].set_xlabel('Scaled ICDF numbers')
    return record_lambda, lambda_error


r_max = 300
lambda_nominal = 45
title = 'Absorption Mean Free Path - Water'
record_lambda_a10, lambda_error10 = Histogram(lambda_nominal, r_max, title)
r_max = 2
title = 'Total Mean Free Path - Water'
record_lambda_w, lambda_eror_w =Histogram(Water()[1], r_max, title)
r_max = 0.1
title = 'Total Mean Free Path - Lithium'
record_lambda_l, lambda_error_l =Histogram(Lithium()[1], r_max, title)
%matplotlib notebook
T = 10
def Neutrons(record_lambda_w, prob_aw, T):
    N = 1
    x_pos = (-record_lambda_w*np.log(np.random.uniform(size = N)))
    y_pos = (-record_lambda_w*np.log(np.random.uniform(size = N)))
    z_pos = (-record_lambda_w*np.log(np.random.uniform(size = N)))
    x = 0
    y = 0
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

        absorb = np.where((np.random.uniform(size = len(x_pos)) < prob_aw))[0]
        x_pos = np.delete(x_pos, absorb, 0)
        y_pos = np.delete(y_pos, absorb, 0)
        z_pos = np.delete(z_pos, absorb, 0)

        a = 1 - np.random.uniform(size = len(x_pos)) * 2
        random_theta = np.arccos(a)
        random_phi = np.random.uniform(size = len(x_pos)) * 2*np.pi
        x_pos += (np.sin(random_theta)*np.cos(random_phi)* (-record_lambda_w*np.log(np.random.uniform())))
        y_pos += (np.sin(random_theta)*np.sin(random_phi)* (-record_lambda_w*np.log(np.random.uniform())))
        z_pos += (np.cos(random_theta)* (-record_lambda_w*np.log(np.random.uniform())))

        x = np.concatenate((x, x_pos))
        y = np.concatenate((y, y_pos))
        z = np.concatenate((z, z_pos))

if len(x) > 0:
        fig = plt.figure(figsize = (15,7))
        ax = fig.add_subplot(1,2,1, projection='3d')
        ax.plot3D(x, y, z, '.')
        ax.plot3D(x, y, z)
        ax.scatter3D(x[0], y[0], z[0], color = 'red', label = 'beginning')
        ax.scatter3D(x[-1], y[-1], z[-1], color = 'blue', label = 'end')
        ax.set_box_aspect(aspect = (1,1,1))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
    else:
        print('no neutrons did anything')
    return

Neutrons(record_lambda_w, prob_aw, T)

%matplotlib notebook
%matplotlib inline
T = 10
def Neutrons(record_lambda, prob_a, T):
    '''
    This function calculates the rate of absorption, reflection and transmission through 10cm of each material.
    '''
    N = 10000
    x_pos = (-record_lambda*np.log(np.random.uniform(size = N)))
    number_transmitted = 0
    number_reflected = 0
    number_absorbed = 0
    while len(x_pos) > 0:
        transmit = np.where((x_pos > T))[0]
        x_pos = np.delete(x_pos, transmit, 0)

        reflect = np.where((x_pos < 0))[0]
        x_pos = np.delete(x_pos, reflect, 0)

        absorb = np.where((np.random.uniform(size = len(x_pos)) < prob_a))[0]
        x_pos = np.delete(x_pos, absorb, 0)

        a = 1 - np.random.uniform(size = len(x_pos)) * 2
        random_theta = np.arccos(a)
        random_phi = np.random.uniform(size = len(x_pos)) * 2*np.pi
        x_pos += (np.sin(random_theta)*np.cos(random_phi)* (-record_lambda*np.log(np.random.uniform(size = len(x_pos)))))

        number_transmitted += len(transmit)
        number_reflected += len(reflect)
        number_absorbed += len(absorb)
    return number_reflected, number_absorbed, number_transmitted

lambda_values = [record_lambda_w, record_lambda_l]
prob_values = [prob_aw, prob_al]
transmit = np.zeros((10 ,2))
absorb = np.zeros((10, 2))
reflect = np.zeros((10, 2))
for i in range(10):
    for j in range(0, len(lambda_values)):
        reflect[i][j], absorb[i][j], transmit[i][j] = Neutrons(lambda_values[j], prob_values[j], T)
mean_reflect = reflect.mean(axis = 0)
mean_absorb = absorb.mean(axis = 0)
mean_transmit = transmit.mean(axis = 0)
std_reflect = reflect.std(axis = 0)
std_absorb = absorb.std(axis = 0)
std_transmit = transmit.std(axis = 0)
materials = 'Water', 'Lithium-6'
processes = 'Reflection', 'Absorption', 'Transmission'
explode = (0.05, 0.05, 0.05)
fig, ax = plt.subplots(1,2,figsize = (12,5))
fig.suptitle('10cm of each material')
fig.tight_layout(pad = 5)
for i in range(0, len(lambda_values)):
    ax[i].set_title(materials[i])
    ax[i].pie((mean_reflect[i], mean_absorb[i], mean_transmit[i]), labels = processes, explode = explode, autopct='%1.0f%%')
    fig, axs = plt.subplots(figsize = (7,1))
    col_labels = [materials[i]]
    row_labels = ['Uncertainty in Reflection%', 'Uncertainty in Absorption%', 'Uncertainty in Transmission%']
    table_vals = [['±{0:.3f}'.format(std_reflect[i]*100/N)], ['±{0:.3f}'.format(std_absorb[i]*100/N)], ['±{0:.3f}'.format(std_transmit[i]*100/N)]]
    axs.axis('off')
    axs.axis('tight')
    axs.table(cellText=table_vals, rowLabels=row_labels, colLabels=col_labels, loc='center')
plt.show()
T = np.arange(0, 20, 1)
def Vary(lambda_v, prob):
    '''
    This function is used to evaluate how each process changes as a function of the thickness of each material.
    '''
    transmit = np.zeros((len(T),10))
    absorb = np.zeros((len(T),10))
    reflect = np.zeros((len(T),10))
    for j in range(10):
        for i in range(0, len(T)):
            reflect[i][j], absorb[i][j], transmit[i][j] = Neutrons(lambda_v, prob, T[i])
    mean_reflect = reflect.mean(axis = 1)
    mean_absorb = absorb.mean(axis = 1)
    mean_transmit = transmit.mean(axis = 1)
    std_reflect = reflect.std(axis = 1)
    std_absorb = absorb.std(axis = 1)
    std_transmit = transmit.std(axis = 1)
    return mean_reflect, mean_absorb, mean_transmit, std_reflect, std_absorb, std_transmit

def Plot():
    fig, ax = plt.subplots(1,2,figsize = (14,5))
    for i in range(0, len(lambda_values)):
        ax[i].set_title(materials[i])
        reflect, absorb, transmit, std_r, std_a, std_t = Vary(lambda_values[i], prob_values[i])
        ax[i].errorbar(T, reflect*100/N, yerr = std_r*100/N, label = 'reflect')
        ax[i].errorbar(T, absorb*100/N, yerr = std_a*100/N, label = 'absorb')
        ax[i].errorbar(T, transmit*100/N, yerr = std_t*100/N, label = 'transmit')
        ax[1].set_xlabel('Thickness/cm')
        ax[0].set_ylabel('Rate of Processes/%')
        ax[i].legend()
    plt.show()
Plot()
`
%matplotlib notebook
record_lambda_1 = record_lambda_l
record_lambda_2 = record_lambda_w
sigmaa = 0
sigma2 = 1/record_lambda_2
sigma1 = 1/record_lambda_1
sigmaT = max(sigma1, sigma2)
T1 = 10
T2 = 10
Well — Today at 12:03 PM
def Wood_cock():
    '''
    This function simulates the motion of a neutron through 10cm vacuum and 10cm water using the Woodcock method.
    '''
    N = 1
    x_pos = (-record_lambda_w*np.log(np.random.uniform(size = N)))
    y_pos = np.zeros(N)
    z_pos = np.zeros(N)
    a = 1 - np.random.uniform(size = len(x_pos)) * 2
    random_theta = np.full(N,np.pi/2)
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
     v = np.random.uniform(size = len(x_pos))
        region2 = np.where((x_pos > T1) & (x_pos <= T1 + T2))
        fic = np.where((v > sigmaa/sigmaT) & (x_pos >= 0) & (x_pos <= T1))
        n_fic = np.where((v <= sigmaa/sigmaT) & (x_pos >= 0) & (x_pos <= T1))

        w = np.random.uniform(size = len(x_pos[fic]))
        x_pos[fic] += (np.sin(random_theta[fic])*np.cos(random_phi[fic])* (-record_lambda_w*np.log(w)))
        y_pos[fic] += (np.sin(random_theta[fic])*np.sin(random_phi[fic])* (-record_lambda_w*np.log(w)))
        z_pos[fic] += (np.cos(random_theta[fic])* (-record_lambda_w*np.log(w)))

        a = 1 - np.random.uniform(size = len(x_pos[n_fic])) * 2
        random_theta[n_fic] = np.arccos(a)
        random_phi[n_fic] = np.random.uniform(size = len(x_pos[n_fic])) * 2*np.pi
        x_pos[n_fic] += (np.sin(random_theta[n_fic])*np.cos(random_phi[n_fic])* (-record_lambda_w*np.log(np.random.uniform(size = len(x_pos[n_fic])))))
        y_pos[n_fic] += (np.sin(random_theta[n_fic])*np.sin(random_phi[n_fic])* (-record_lambda_w*np.log(np.random.uniform(size = len(x_pos[n_fic])))))
        z_pos[n_fic] += (np.cos(random_theta[n_fic])* (-record_lambda_w*np.log(np.random.uniform(size = len(x_pos[n_fic])))))
   a = 1 - np.random.uniform(size = len(x_pos[region2])) * 2
        random_theta[region2] = np.arccos(a)
        random_phi[region2] = np.random.uniform(size = len(x_pos[region2])) * 2*np.pi
        x_pos[region2] += (np.sin(random_theta[region2])*np.cos(random_phi[region2])* (-record_lambda_w*np.log(np.random.uniform(size = len(x_pos[region2])))))
        y_pos[region2] += (np.sin(random_theta[region2])*np.sin(random_phi[region2])* (-record_lambda_w*np.log(np.random.uniform(size = len(x_pos[region2])))))
        z_pos[region2] += (np.sin(random_theta[region2])* (-record_lambda_w*np.log(np.random.uniform(size = len(x_pos[region2])))))

        absorb1 = np.where((np.random.uniform(size = len(x_pos)) < prob_aw) & (x_pos > T1) & (x_pos <= T1 + T2))[0]
        x_pos = np.delete(x_pos, absorb1, 0)
        y_pos = np.delete(y_pos, absorb1, 0)
        z_pos = np.delete(z_pos, absorb1, 0)
        random_theta = np.delete(random_theta, absorb1, 0)
        random_phi = np.delete(random_phi, absorb1, 0)
        '''
        absorb2 = np.where((np.random.uniform(size = len(x_pos)) <= sigma1/sigmaT) & (np.random.uniform(size = len(x_pos)) < 0) & (x_pos < T1) & (x_pos > 0))[0]
        x_pos = np.delete(x_pos, absorb2, 0)
        y_pos = np.delete(y_pos, absorb2, 0)
        z_pos = np.delete(z_pos, absorb2, 0)
        random_theta = np.delete(random_theta, absorb2, 0)
        random_phi = np.delete(random_phi, absorb2, 0)
        '''
        x = np.concatenate((x, x_pos))
        y = np.concatenate((y, y_pos))
        z = np.concatenate((z, z_pos))

    fig = plt.figure(figsize = (15,7))
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot3D(x, y, z, '.')
    ax.plot3D(x, y, z)
    ax.scatter3D(x[0], y[0], z[0], color = 'red', label = 'beginning')
    ax.scatter3D(x[-1], y[-1], z[-1], color = 'blue', label = 'end')
    ax.set_box_aspect(aspect = (1,1,1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    return
Wood_cock()