"""
This script creates two main outputs:
1) All the available tc ages within the grid of the models;
2) All the avaialble tc ages within the grid of the models
with uncertainties assigned that include the spatial variability
of the observations

=============================================
Requirements:
    *
=============================================

VUW
July 2019
Author: Konstantinos Michailos
"""

import numpy as np
import csv
import os
from shapely.geometry import Polygon, Point

# Define working directory
work_dir = os.getcwd()

def read_csv_file(path, num_of_columns, delimitah=','):
    """Return a number of lists (number of columns)."""
    details_file = path

    lists = [[] for i in range(num_of_columns)]
    for i in range(num_of_columns):
        print(i)
        c_reader = csv.reader(open(details_file, 'r'), delimiter=delimitah)
        lists[i] = list(zip(*c_reader))[i]
    return lists


def dist_calc(loc1, loc2):
    """
    Function to calculate the distance in km between two points.

    Uses the flat Earth approximation. Better things are available for this,
    like `gdal <http://www.gdal.org/>`_.

    :type loc1: tuple
    :param loc1: Tuple of lat, lon, depth (in decimal degrees and km)
    :type loc2: tuple
    :param loc2: Tuple of lat, lon, depth (in decimal degrees and km)

    :returns: Distance between points in km.
    :rtype: float
    :author: Calum Chamberlain
    """
    R = 6371.009  # Radius of the Earth in km
    dlat = np.radians(abs(loc1[0] - loc2[0]))
    dlong = np.radians(abs(loc1[1] - loc2[1]))
    ddepth = abs(loc1[2] - loc2[2])
    mean_lat = np.radians((loc1[0] + loc2[0]) / 2)
    dist = R * np.sqrt(dlat ** 2 + (np.cos(mean_lat) * dlong) ** 2)
    dist = np.sqrt(dist ** 2 + ddepth ** 2)
    return dist


######################
# Read thermochron obs
# """"""""""""""""""""

# Read all available thermochron obs
Tc_path = (work_dir)

FT_file = (Tc_path + '/All_thermochron_data.csv')
# Read parameters from .csv file
params = read_csv_file(FT_file, 6)
# Read bin corners
X_FT = [float(i) for i in params[0]]    # lon 
Y_FT = [float(i) for i in params[1]]    # lat
Age_FT = [float(i) for i in params[2]]  # Ages(Myr)
Err_FT_ = [float(i) for i in params[3]] # Errors
Tc_FT = [float(i) for i in params[4]]   # Closing temp
El_FT = [float(i) for i in params[5]]   # Elevation (m)
Err_FT = []
for i, j in enumerate(Err_FT_):
    error = j
    if error == 0.0:
        error = 0.1
    Err_FT.append(error)
# Keep observations within the box
# defined by the following coordinates
# plats = [-43.58, -43.11, -43.41, -43.95, -43.58]
# plons = [169.89, 170.85, 171.15, 170.19, 169.89]
plats = [-44, -42.85, -43.225, -44.375]
plons = [168.95, 171.25, 171.55, 169.25]
p_ = Polygon((np.asarray(list(zip(plats, plons)))))
inside_box = np.array([[X_FT[i], Y_FT[i], Age_FT[i], Err_FT[i],
                        Tc_FT[i], El_FT[i]]
                        for i in range(len(Y_FT))
                        if p_.contains(Point(Y_FT[i], X_FT[i]))])
# List of the ZFT data details
x_FT = []
y_FT = []
age_FT = []
err_FT = []
tc_FT = []
el_FT = []
for i, j in enumerate(inside_box):
    # Age = j[2]
    if j[2] < 10.0:
        # We dont need those
        x_FT.append(j[0])
        y_FT.append(j[1])
        age_FT.append(j[2])
        error = j[3]
        if error == 0.0:
            error = 0.1
        err_FT.append(error)
        tc_FT.append(j[4])
        el_FT.append(j[5])

thermocron_obs = np.array([[x_FT[i], y_FT[i], age_FT[i], err_FT[i], 
                            tc_FT[i], el_FT[i]]
                            for i in range(len(y_FT))])

# Write the first output (1)
for i, j in enumerate(thermocron_obs):
    with open(work_dir+
                '/Thermochron_data_in_grid.csv', 'a') as of:
        of.write('{}, {}, {}, {}, {}, {}\n'.
                format(j[0], j[1], j[2], j[3], j[4], j[5]))





file_path = (Tc_path + '/Thermochron_data_in_grid.csv')

params = read_csv_file(file_path, 6)

lon_FT = [float(i) for i in params[0]]    # lon
lat_FT = [float(i) for i in params[1]]    # lat
Age_FT = [float(i) for i in params[2]]    # Ages(Myr)
Err_FT = [float(i) for i in params[3]]   # Errors
El_FT = [float(i) for i in params[5]]   # Elevation (m)
Tc_FT = [float(i) for i in params[4]]   # Closure temperature

loc_array = []
for i, j in enumerate(lon_FT):
    loc_array.append([lon_FT[i], lat_FT[i], 10])

radius = 20

for i, datum in enumerate(lon_FT):
    # print i
    ages_nearby = []
    for j, loc in enumerate(loc_array):
        dist = dist_calc(loc, [lon_FT[i],lat_FT[i],10])
        if dist < radius/2:
            ages_nearby.append(Age_FT[j])
            # print dist, Age_FT[i], Age_FT[j]
    # May 2020 add np.std was /n adding ddof=1 becomes /n-1
    sigma_latin = np.std(ages_nearby,ddof=1)
    # sigma_latin = np.std(ages_nearby)
    #print sigma_latin, Age_FT[i]
    sigma_total = np.sqrt(sigma_latin**2 + Err_FT[i]**2)
    print(lon_FT[i], lat_FT[i], Age_FT[i],round(sigma_total,1))
    with open(work_dir+'/Tc_unc.csv', 'a') as f:
        f.write('{}, {}, {}, {}, {}, {}\n'. format(lon_FT[i], lat_FT[i],
                Age_FT[i], round(sigma_total, 1), Tc_FT[i], El_FT[i]))