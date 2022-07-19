"""
List of functions to estimate exhumation rates
along the length of the Alpine Fault using a
steady state 1-D temperature model with a number
and a number of boundary conditions and a number
of different observations (seismicity depths,
thermochron ages, geodetic uplifts) to constrain
the model and calculate exhumation rates.

=============================================
Requirements:
    *
    *
=============================================

VUW
May 2019
Author: Konstantinos Michailos
"""
import numpy as np
from scipy.optimize import minimize
import os
from main_functions.functions import *

# Define fixed parameters
# Year in seconds
year = 365.25*24*60*60
# Material properties
# average crustal density of 2700 kg m-3
# J/kgK Volumetric heat capacity = density * specific H capacity
C = 2700.*790.
# W/mK Thermal conductivity
k = 3.2
# W/m^3s Volumetric heat productivity - typical value for granite/
# heat production
H = 3.e-6
# Sea level; top of exhuming block
z0 = 0.
# Temperature at the top of exhuming block
T0 = 13.
# Regularization parameter (Q = alpha * Q_therm + Q_seis)
alpha = 1.0
# Decollement depth; bottom of exhuming block
z1 = -35000.
# Temperature at the bottom of exhuming block
T1 = 550.
# Choose which model to run
# Steady state exhumation OR initial geotherm
assumption = '_steady_state'
# assumption = '_initial_state'

# Define working directory
work_dir = os.getcwd()
# Define path of thermochronological data
Tc_path = (work_dir + '/dataset/tc_data/')
# Define path of seismicity data
npzfilespath = (work_dir + '/dataset/npz_files/')
# Read them observations
observations = read_observations(Tc_path, npzfilespath)
seis_obs = observations[1]
thermochron_obs = observations[0]

# Initial guesses for adjustable parameters
# Temperature at the brittle-ductile depth (initial guess)
T_BDT = 450.
# First line of grid points
vel_A1 = 2.0
vel_A2 = 2.0
vel_A3 = 2.
vel_A4 = 4.
vel_A5 = 8.
vel_A6 = 6.5
vel_A7 = 4.
vel_A8 = 2.
vel_A9 = 1.5
vel_A10 = 1.
vel_A11 = 1.
# Second line of grid points
vel_B1 = 1.5
vel_B2 = 1.5
vel_B3 = 1.5
vel_B4 = 3.
vel_B5 = 6.
vel_B6 = 6.
vel_B7 = 4.
vel_B8 = 2.
vel_B9 = 2.
vel_B10 = 1.
vel_B11 = 1.
# Third line of grid points
vel_C1 = 1.5
vel_C2 = 1.5
vel_C3 = 1.5
vel_C4 = 2.
vel_C5 = 4.
vel_C6 = 4.
vel_C7 = 3.
vel_C8 = 2.
vel_C9 = 2.
vel_C10 = 1.
vel_C11 = 1.

# Set model's run name to be used
# for creating a folder in the oudputs directory
run_name = ('Tbdt_' + str(int(T_BDT)) + '_T1_' +
            str(int(T1)) + '_' + assumption)

# Parameters for the minimization process
# List of adjustable parameters
# (note that z1 and T1 are in the adjustable but are used as fixed)
p = [vel_A1, vel_A2, vel_A3, vel_A4, vel_A5,
     vel_A6, vel_A7, vel_A8, vel_A9,
     vel_A10, vel_A11,
     vel_B1, vel_B2, vel_B3, vel_B4, vel_B5,
     vel_B6, vel_B7, vel_B8, vel_B9,
     vel_B10, vel_B11,
     vel_C1, vel_C2, vel_C3, vel_C4, vel_C5,
     vel_C6, vel_C7, vel_C8, vel_C9,
     vel_C10, vel_C11, z1, T1, T_BDT]
# List of fixed parameters
fixed_par = [T1, z1, thermochron_obs, seis_obs, alpha, work_dir, run_name]
pfix = np.array(fixed_par)
x0 = np.array(p)

# Make directories for outputs of model
output_path = os.path.join(work_dir, 'outputs', run_name)
make_dirs(output_path)
log_dir = os.path.join(output_path, 'log_files')
make_dirs(log_dir)
plot_dir = os.path.join(output_path, 'plots')
make_dirs(plot_dir)

# If these directories exist delete the files within em
filenames = [log_dir + '/depths.csv',
             log_dir + '/ages.csv',
             output_path + '/mod_uplifts.txt',
             log_dir + '/thermo_res.csv',
             log_dir + '/seis_res.csv',
             log_dir + '/temps.txt',
             log_dir + '/logfile_SWISS.txt']

for filepath in filenames:
    try:
        os.remove(filepath)
    except Exception as e:
        print(e)
        print("Error while deleting file ", filepath, ", file doesn't exist!")

def printx(Xi):
    """Function used to output the values of the parameters as the iterations
       progress.
    """
    global Nfeval
    global fout
    # print(Nfeval, Q(Xi))
    fout.write('{0: 3.1f}, {1: 3.1f}, {2: 3.1f},{3: 3.1f}, {4: 3.1f}, {5: 3.1f},\
                {6: 3.1f}, {7: 3.1f}, {8: 3.1f},{9: 3.1f},\
                {10: 3.1f}, {11: 3.1f}, {12: 3.1f},{13: 3.1f}, {14: 3.1f}, {15: 3.1f},\
                {16: 3.1f}, {17: 3.1f}, {18: 3.1f},{19: 3.1f},{20: 3.1f}, {21: 3.1f}, {22: 3.1f},{23: 3.1f}, {24: 3.1f}, {25: 3.1f},\
                {26: 3.1f}, {27: 3.1f}, {28: 3.1f},{29: 3.1f}, {30: 3.1f}, {31: 3.1f},\
                {32: 3.1f}, {33: 3.1f}, {34:4d}, {35: 5.2f}'
                 .format(Xi[0], Xi[1], Xi[2],
                         Xi[3], Xi[4], Xi[5],
                         Xi[6], Xi[7], Xi[8],
                         Xi[9], Xi[10], Xi[11],
                         Xi[12], Xi[13], Xi[14],
                         Xi[15], Xi[16], Xi[17],
                         Xi[18], Xi[19], Xi[20],
                         Xi[21], Xi[22], Xi[23],
                         Xi[24], Xi[25], Xi[26],
                         Xi[27], Xi[28], Xi[29],
                         Xi[30], Xi[31], Xi[32], Xi[33],
                         Nfeval, Q(Xi, pfix)) + '\n')
    Nfeval += 1


Nfeval = 1
fout = open(log_dir + '/logfile_SWISS.txt', 'w')

# Choosing which Q function to use from functions.py
if assumption == '_steady_state':
    Q = Q_steady_state
else:
    Q = Q_initial_state
# Run minimization process
res = minimize(Q, x0, args=(pfix), method='powell',
               callback=printx, options={'disp': True, 'maxiter': 200})
fout.close()

# Read minimization process results and set the final modelled
# exhumation rates on the model's grid points
final_uplifts = define_grid(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4],
                            res.x[5], res.x[6], res.x[7], res.x[8], res.x[9],
                            res.x[10], res.x[11], res.x[12], res.x[13],
                            res.x[14], res.x[15], res.x[16], res.x[17],
                            res.x[18], res.x[19], res.x[20], res.x[21],
                            res.x[22], res.x[23], res.x[24], res.x[25],
                            res.x[26], res.x[27], res.x[28], res.x[29],
                            res.x[30], res.x[31], res.x[32])

# Write output for with final exhumation rates
for i, j in enumerate(final_uplifts[0]):
    print(final_uplifts[1][i])
    with open(output_path + '/mod_uplifts.txt', 'a') as of:
        of.write('{} {} {}\n'.
                 format(final_uplifts[1][i],
                        final_uplifts[0][i],
                        round(final_uplifts[2][i], 2)))

# Make plots
# Read residuals from the last iteration of the minimization
# and create lists that will be inputs for the maps
seis_res = sort_residuals('seis_res.csv', log_dir)
ther_res = sort_residuals('thermo_res.csv', log_dir)
plot3 = plot_res(seis_res, ther_res, run_name, plot_dir)
plots1 = plot_temps(final_uplifts, run_name, p, res, work_dir, plot_dir)
plots2 = plot_exhum(final_uplifts, run_name, plot_dir)
# This line creates the input for the gmt script cross_sections_temperat.sh
cross_section_prep = temp_cross_section(final_uplifts, p, work_dir)
