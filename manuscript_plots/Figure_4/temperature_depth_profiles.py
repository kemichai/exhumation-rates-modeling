"""
Script that plots the temperature vs depth profiles for various
exhumation values and from a couple of previous studies.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def exhumeT(z, v, C=2700.*790., k=3.5, H=3.e-6, z0=0.,
            T0=0., z1=-50000., T1=550.):
    """
    Computes a steady state T profile for constant exhumation rate and
    fixed boundary conditions.
    ARGS:
        z       value to compute T at
        v       exhumation velocity
        C       volumetric heat capacity
        k,H     thermal conductivity, volumetric heat productivity
        z0,T0   top boundary condition
        z1,T1   base boundary condition
    RETURNS:
        T       array of temperatures for each depth in zArray
    """
    A2 = ((T1-T0) - (z1-z0)*H/(v*C)) / (np.exp(z1*v*C/k)-np.exp(z0*v*C/k))
    A1 = T0 - A2*np.exp(z0*v*C/k) - z0*H/(v*C)

    return A1 + A2*np.exp(z*v*C/k) + z*H/(v*C)


year = 365.25*24*60*60
# Material properties
# J/kgK Volumetric heat capacity = density * specific H capacity
C = 2700.*790.
# W/mk Thermal conductivity
k = 3.2
#
H = 2.e-6
# Exhumation velocities
# velocity in mm/yr converted to m/s
v_0 = 0.0001e-3/year
v_1 = 1.e-3/year
v_2 = 2.1e-3/year
v_3 = 4.1e-3/year
v_4 = 6.1e-3/year
v_5 = 8.1e-3/year

# Boundary Conditions
z0 = 0.
T0 = 13.
z1 = -35000.
T1 = 550.

# Define working directory
working_dir = os.getcwd()

###########################################
z_ = np.linspace(-3000, -32000., 10)
T_0_ = exhumeT(z_, v_0, C, k, H, z0, T0, z1, T1)
T_1_ = exhumeT(z_, v_1, C, k, H, z0, T0, z1, T1)
T_2_ = exhumeT(z_, v_2, C, k, H, z0, T0, z1, T1)
T_3_ = exhumeT(z_, v_3, C, k, H, z0, T0, z1, T1)
T_4_ = exhumeT(z_, v_4, C, k, H, z0, T0, z1, T1)
T_5_ = exhumeT(z_, v_5, C, k, H, z0, T0, z1, T1)
z_list = z_.tolist()
z_km_ = [-1*float(i)/1000 for i in z_list]
###########################################
z = np.linspace(z0, z1, 100)
T_0 = exhumeT(z, v_0, C, k, H, z0, T0, z1, T1)
T_1 = exhumeT(z, v_1, C, k, H, z0, T0, z1, T1)
T_2 = exhumeT(z, v_2, C, k, H, z0, T0, z1, T1)
T_3 = exhumeT(z, v_3, C, k, H, z0, T0, z1, T1)
T_4 = exhumeT(z, v_4, C, k, H, z0, T0, z1, T1)
T_5 = exhumeT(z, v_5, C, k, H, z0, T0, z1, T1)
z_list = z.tolist()
z_km = [-1*float(i)/1000 for i in z_list]

# Previous studies of Toy et al 2010 and Cross et al. 2015
T_toy10 = [0.0, 300, 650]
d_toy10 = [0.0, 7.0, 35]
T_cross15 = [0.0, 500, 650]
d_cross15 = [0.0, 11.0, 35.0]

# Temperaturevs. depth profiles as a function of time in 1 Myr intervals.
# The first row of the txt file is the depth scale in m.
# The second row is the stable geotherm (deg. C),
# the third is after 1 Myr, 4 after 2 Myr, etc.

text_file = working_dir + '/geotherms.txt'
array = np.loadtxt(text_file, delimiter=',')

depths = array[0]
depths_km = [-1*float(i)/1000 for i in depths]

stable_geotherm = array[1]
one_myr = array[2]
two_myr = array[3]
three_myr = array[4]
four_myr = array[5]
five_myr = array[6]
six_myr = array[7]
sev_myr = array[8]
eig_myr = array[9]
nin_myr = array[10]
steady_state = array[11]


################################
# Plotting section of the script
################################
font = {'family': 'normal',
        'weight': 'normal',
        'size': 18}
matplotlib.rc('font', **font)
# Set figure width to 12 and height to 9
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
################################
ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
ax1.plot(T_toy10, d_toy10, zorder=1, color='grey',
         linestyle='dashed', label='Toy et al. (2010)')
ax1.plot(T_cross15, d_cross15, zorder=1, color='grey',
         linestyle='dotted', label='Cross et al. (2015)')
ax1.plot(T_0, z_km, zorder=2, color='k', linestyle='solid',
         label='0.0 mm/yr')
ax1.scatter(T_0_, z_km_, facecolor='white', alpha=1,
            edgecolor='k', linewidth=1., zorder=3)
ax1.plot(T_2, z_km, zorder=2, color='k', linestyle='solid', label='2.0 mm/yr')
ax1.scatter(T_2_, z_km_, facecolor='white', alpha=1, edgecolor='k', marker='D',
            linewidth=1., zorder=3)
ax1.plot(T_3, z_km, zorder=2, color='k', linestyle='solid', label='4.0 mm/yr')
ax1.scatter(T_3_, z_km_, facecolor='white', alpha=1, edgecolor='k', marker='^',
            linewidth=1., zorder=3)
ax1.plot(T_4, z_km, zorder=2, color='k', linestyle='solid', label='6.0 mm/yr')
ax1.scatter(T_4_, z_km_, facecolor='white', alpha=1, edgecolor='k', marker='s',
            linewidth=1., zorder=3)
ax1.plot(T_5, z_km, zorder=2, color='k', linestyle='solid', label='8.0 mm/yr')
ax1.scatter(T_5_, z_km_, facecolor='white', alpha=1, edgecolor='k', marker='v',
            linewidth=1., zorder=3)
ax1.set_ylabel('Depth (km)', fontsize=18)
ax1.set_xlabel(u'Temperature (℃)', fontsize=18)
ax1.scatter(13, 0, facecolor='grey', alpha=1, edgecolor='black',
            linewidth=1., zorder=3)
ax1.scatter(550, 35, facecolor='black', alpha=1, edgecolor='black',
            linewidth=1., zorder=3)
ax1.set_ylim([-0.3, 36])
ax1.set_xlim([-3, 650])
plt.gca().invert_yaxis()
plt.legend(loc="lower left", markerscale=1., scatterpoints=1, fontsize=14)
# plt.suptitle('Estimated temperature profile for the central Southern Alps',
# fontsize=20)
#
ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)
ax2.plot(stable_geotherm, depths_km, zorder=1, color='k', linestyle='solid',
         linewidth=2)
ax2.plot(one_myr, depths_km, zorder=2, color='grey', linestyle='--')
ax2.plot(two_myr, depths_km, zorder=2, color='grey', linestyle='--')
ax2.plot(three_myr, depths_km, zorder=2, color='grey', linestyle='--')
ax2.plot(four_myr, depths_km, zorder=2,  color='grey', linestyle='--')
ax2.plot(five_myr, depths_km, zorder=2,  color='grey', linestyle='--')
ax2.plot(six_myr, depths_km, zorder=2,  color='grey', linestyle='--')
ax2.plot(sev_myr, depths_km, zorder=2,  color='grey', linestyle='--')
ax2.plot(eig_myr, depths_km, zorder=2,  color='grey', linestyle='--')
ax2.plot(nin_myr, depths_km, zorder=2,  color='grey', linestyle='--')
ax2.plot(steady_state, depths_km, zorder=2, color='k', linestyle='solid',
         linewidth=2)
ax2.set_ylim([-0.3, 36])
ax2.set_xlim([-3, 650])
plt.gca().invert_yaxis()
# plt.legend(loc="lower left", markerscale=1., scatterpoints=1, fontsize=14,
#            ncol=2)
ax2.set_xlabel(u'Temperature (℃)', fontsize=18)
fig_name = working_dir + '/Fig_4.png'
plt.savefig(fig_name, bbox_inches="tight", format='png')
plt.show()
