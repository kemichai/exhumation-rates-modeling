#!/usr/bin/env python.
"""
Script that reads an earthquake catalog and populates the locations into a
given set of boxes/bins defined using quadtree algorithm (MATLAB codes).

Te Whare wananga o te upoko o te Ika a Maui
VUW
February 2019
Author: Konstantinos Michailos
"""

import csv
from pyproj import Proj, transform
import numpy as np

# class used to rotate coordinates
class CoordRotator:
    def __init__(self, origin, angle):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.angle = float(angle)

    def forward(self, x, y):
        rx = np.asarray(x, dtype=np.float64) - self.origin[0]
        ry = np.asarray(y, dtype=np.float64) - self.origin[1]
        ca = np.cos(self.angle)
        sa = np.sin(self.angle)
        xx = rx * ca + ry * sa
        yy = -rx * sa + ry * ca
        return xx, yy

    def inverse(self, xx, yy, z=None):
        ca = np.cos(self.angle)
        sa = np.sin(self.angle)
        rx = xx * ca + -yy * sa
        ry = xx * sa + yy * ca
        x = rx + self.origin[0]
        y = ry + self.origin[1]
        return x, y


degrees = -54
inProject = Proj(init='epsg:4326')
outProject = Proj(init='epsg:2193')

###############################################################################
# Create input for MATLAB codes
###############################################################################
# Read the csv file that contains the vertical uncertainties
working_dir = "/home/kostas/Dropbox/VUW/PhD/Codes/Modeling_1/"
file_name = 'dataset_all.csv'
# File dataset_all has the median uncertainty for the depth assigned to the
# locations that do not have an uncertainty calculation
csv_file = working_dir + file_name
# Read parameters from .csv file
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
lat = list(zip(*c_reader))[6]
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
lon = list(zip(*c_reader))[7]
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
dep = list(zip(*c_reader))[8]
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
dep_unc = list(zip(*c_reader))[11]
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
# Earthquake catalog details
# Could add the time for matching here...
lat = [float(i) for i in lat]
lon = [float(i) for i in lon]
dep = [float(i) for i in dep]
dep_unc = [float(i) for i in dep_unc]

outpath = ('/Volumes/GeoPhysics_05/users-data/michaiko/'
           'Matlab/Dep_distr/Inputs/')

# Define the projections
inProj = inProject
outProj = outProject

# Convert from lat/lon to NZGD2000 (New Zealand Transverse Mercator 2000)
nz_x, nz_y, z = transform(inProj, outProj, lon, lat, dep)

# Define origin point
orig = [0, 0]
x2_ = np.asarray(nz_x) + orig[0]
y2_ = np.asarray(nz_y) + orig[1]

aaa = CoordRotator(orig, np.radians(-54))
nz_x_rot, nz_y_rot = aaa.forward(x2_, y2_)

# Keep eqz with depths smaller than 25 km
for i, j in enumerate(dep):
    if j <= 60:
        print nz_x_rot[i], nz_y_rot[i], j, i
        with open(outpath + 'mat_input_lat_lon_dep.dat', 'a') as f:
                f.write('{} {} {} {} {} {} {} \n'.format(nz_x_rot[i],
                        nz_y_rot[i], j, lat[i], lon[i], dep[i],
                        dep_unc[i]))
###############################################################################
# Run matlab codes
###############################################################################
