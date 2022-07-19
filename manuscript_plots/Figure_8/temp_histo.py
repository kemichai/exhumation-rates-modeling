"""
Script that makes a number of plots.

Attention: the plots below will only work after running the codes in
           Seism_and_thermo.py

"""
import numpy as np
import matplotlib
import os
from main_functions.functions import *
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")


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


def eq_temp(vel_,z_):
    """Use the interpolated exhumation rate (vel) and the earthquake's
       hypocentral depth to calculate the temperature of the individual
       earthquake.
    """
    vel = vel_/year/1000
    z = z_*(-1000)
    A2 = ((T1-T0) - (z1-z0)*H/(vel*C)) / (np.exp(z1*vel*C/k)-np.exp(z0*vel*C/k))
    A1 = T0 - A2*np.exp(z0*vel*C/k) - z0*H/(vel*C)

    return A1 + A2*np.exp(z*vel*C/k) + z*H/(vel*C)


# Define working directory
working_dir = os.getcwd()
# Read exhumation rates from model assuming
# steady-state exhumation geotherm
file_name = '/mod_uplifts_alpha1.txt'
with open(working_dir + file_name, 'r') as f:
    upl_a = []
    lon_a = []
    lat_a = []
    for i, line in enumerate(f):
        print(line)
        ln = line.split()
        lon = float(ln[0])
        lat = float(ln[1])
        upl = float(ln[2])

        lon_a.append(lon)
        lat_a.append(lat)
        upl_a.append(upl)

interpolated_surf_a = interp_surf(lon_a, lat_a, upl_a)

# Read the individual earthquake locations and in particular
# the csv file that contains the vertical uncertainties
file_name = '/dataset_all.csv'
eq_cat_file = working_dir + file_name
# Define paramters to be used in the calculations
params = read_csv_file(eq_cat_file, 12)
eq_lon = [float(i) for i in params[7]]
eq_lat = [float(i) for i in params[6]]
eq_dep = [float(i) for i in params[8]]

earthquake_temp_a = []
earthquake_dep_a = []
for i, j in enumerate(eq_lon):
    exh = interpolated_surf_a(eq_lon[i], eq_lat[i])
    eq_t = eq_temp(exh[0], eq_dep[i])
    earthquake_temp_a.append(eq_t)
    earthquake_dep_a.append(eq_dep[i])

# Read exhumation rates from model assuming
# initial stable geotherm with non-exhuming crust
file_name = '/mod_uplifts_init.txt'
with open(working_dir + file_name, 'r') as f:
    upl_b = []
    lon_b = []
    lat_b = []
    for i, line in enumerate(f):
        print(line)
        ln = line.split()
        lon = float(ln[0])
        lat = float(ln[1])
        upl = float(ln[2])

        lon_b.append(lon)
        lat_b.append(lat)
        upl_b.append(upl)

interpolated_surf_b = interp_surf(lon_b, lat_b, upl_b)

earthquake_temp = []
earthquake_dep = []
for i, j in enumerate(eq_lon):
    exh = interpolated_surf_b(eq_lon[i], eq_lat[i])
    eq_t = eq_temp(exh[0], eq_dep[i])
    earthquake_temp.append(eq_t)
    earthquake_dep.append(eq_dep[i])

# Plots
font = {'family': 'normal',
        'weight': 'normal',
        'size': 18}
matplotlib.rc('font', **font)
# Set figure width to 12 and height to 9
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size
################################
################################
bins = np.arange(-5, 655, 10)
ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
ax1.hist(earthquake_temp, bins, histtype='step', orientation='vertical',
         color='black', facecolor='grey', alpha=0.7, linewidth=1.5,
         edgecolor='k', fill=True, label='Initial geotherm')
ax1.hist(earthquake_temp_a, bins, histtype='step', orientation='vertical',
         color='black', facecolor='grey', alpha=0.7, linewidth=1.5,
         edgecolor='red', fill=True, linestyle='--', label='Steady-state; a=1')
ax1.set_xlim([0, 650])
ax1.set_ylim([0, 500])
ax1.set_xlabel(u'Temperature (℃)', fontsize=18)
ax1.set_ylabel(r'Number of events', fontsize=18)
plt.axvline(483.4, color='red', linestyle='--', linewidth=2,
            label=r'T$_{BDT}$ (' + str(483) + u'℃)')
ax1.axvline(461.2, 0, 90, lw=2, color='k', linestyle='-',
            label=r'T$_{BDT}$ (' + str(461) + u'℃)')
ax1.legend(loc="upper left", markerscale=1., scatterpoints=1,
           fontsize=17, framealpha=1, borderpad=1)
# ax1.axvline(300, 0, 90, lw=1.5, color='k',linestyle='--')
# ax1.axvline(450, 0, 90, lw=1.5, color='k',linestyle='--')
# plt.text(300 +75, 250, 'Greenschist',
#          {'color': 'black', 'fontsize': 12, 'ha': 'center', 'va': 'center',
#           'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
# plt.text(460 + 75, 250, 'Amphibolite',
#          {'color': 'black', 'fontsize': 12, 'ha': 'center', 'va': 'center',
#           'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
# plt.text(50 +75, 250, 'Pumpellyite-Actinolite',
#          {'color': 'black', 'fontsize': 12, 'ha': 'center', 'va': 'center',
#           'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
# plt.text(0 +150, 250, 'Zeolite',
#          {'color': 'black', 'fontsize': 12, 'ha': 'center', 'va': 'center',
#           'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
ax1.yaxis.set_label_position('right')
ax1.yaxis.tick_right()
fig_name = working_dir + '/Fig_8.png'
plt.savefig(fig_name, bbox_inches="tight", format='png')

plt.show()
