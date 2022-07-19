"""
- Read earthquake catalog 
- Create the inputs for the matlab quadtree codes
=============================================
Requirements:
    *  
=============================================

VUW
May 2019
Author: Konstantinos Michailos
"""
import csv
from pyproj import Proj, transform
import numpy as np

from functions import read_csv_file
from functions import CoordRotator

# Define parameters for coordinates rotation and transformation
degrees = -54
inProject = Proj(init='epsg:4326')
outProject = Proj(init='epsg:2193')

# Define path of the csv file that contains the earthquake catalog
working_dir = "/home/michaiko/Dropbox/Dropbox/VUW/PhD/Codes/Modeling_1/"
file_name = 'dataset_all.csv'
# File dataset_all has the median uncertainty for the depth assigned to the
# locations that do not have an uncertainty calculation

# Read parameters from .csv file
csv_file = working_dir + file_name
params = read_csv_file(csv_file, 12)
# Create lists
lat = [float(i) for i in params[6]]
lon = [float(i) for i in params[7]]
dep = [float(i) for i in params[8]]
dep_unc = [float(i) for i in params[11]]

# Define output path (input for Matlab codes)
outpath = ('/Volumes/GeoPhysics_05/users-data/michaiko/'
           'Matlab/Dep_distr/Inputs/')

# Convert from lat/lon to NZGD2000 (New Zealand Transverse Mercator 2000)
nz_x, nz_y, z = transform(inProject, outProject, lon, lat, dep)

# Define origin point
orig = [0, 0]
x2_ = np.asarray(nz_x) + orig[0]
y2_ = np.asarray(nz_y) + orig[1]

aaa = CoordRotator(orig, np.radians(-54))
nz_x_rot, nz_y_rot = aaa.forward(x2_, y2_)

# Keep eqz with depths smaller than ... km
for i, j in enumerate(dep):
    if j <= 60:
        print nz_x_rot[i], nz_y_rot[i], j, i
        with open(outpath + 'mat_input_lat_lon_dep.dat', 'a') as f:
                f.write('{} {} {} {} {} {} {} \n'.format(nz_x_rot[i],
                        nz_y_rot[i], j, lat[i], lon[i], dep[i],
                        dep_unc[i]))