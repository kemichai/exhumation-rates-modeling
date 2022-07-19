"""
Script that reads thermochron data, clusters them using a number of 
different methods and finally calculates the new sigma that includes 
the spatial variability of the ages...

=============================================
Requirements:
    *
=============================================

VUW
July 2019
Author: Konstantinos Michailos
"""
from obspy import read_events
from sklearn.cluster import KMeans
from obspy import Catalog
from itertools import cycle
from obspy.geodetics import degrees2kilometers, kilometer2degrees
import csv
from pyproj import Proj, transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools


def read_csv_file(path, num_of_columns, delimitah=','):
    """Return a number of lists (number of columns)."""
    details_file = path

    lists = [[] for i in xrange(num_of_columns)]
    for i in range(num_of_columns):
        print i
        c_reader = csv.reader(open(details_file, 'r'), delimiter=delimitah)
        lists[i] = list(zip(*c_reader))[i]
    return lists


######################
# Read thermochron obs
# """"""""""""""""""""
Tc_path = ('/home/michaiko/Dropbox/Dropbox/VUW/PhD/Codes/'
           'Modeling_1/Latest_codes/modelling_codes/tc_data/')
file_path = (Tc_path + 'Thermochron_data_in_grid.csv')

params = read_csv_file(file_path, 6)

lon_FT = [float(i) for i in params[0]]    # lon
lat_FT = [float(i) for i in params[1]]    # lat
Age_FT = [float(i) for i in params[2]]    # Ages(Myr)
Err_FT = [float(i) for i in params[3]]   # Errors
El_FT = [float(i) for i in params[5]]   # Elevation (m)
Tc_FT = [float(i) for i in params[4]]   # Closure temperature
# meth_FT = params[5]

dat_array = []
# Populate it
for i, j in enumerate(lon_FT):
    dat_array.append([lon_FT[i], lat_FT[i], Age_FT[i], Err_FT[i], El_FT[i], Tc_FT[i]])


# import matplotlib.pyplot as plt
# ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
# ax1.scatter(lon_FT, lat_FT, color='orange', marker='o',
#             edgecolor='black', alpha=1,
#             label='Grid points')
# plt.show()


######################
# kmeans
# """"""""""""""""""""
# Generate sample data
# Use kmeans to divide study area in 3 sub-regions
n_clusters = 15
# Make the location array
loc_array = []
# Populate it
for i, j in enumerate(dat_array):
    loc_array.append([dat_array[i][0], dat_array[i][1], 10])
# Run kmeans algorithm
kmeans = KMeans(n_clusters=n_clusters).fit(loc_array)

# Get group index for each event
indices = kmeans.fit_predict(loc_array)
# Preallocate group catalogs
group_dat = [[] for i in range(n_clusters)]
for i, obs in enumerate(dat_array):
    group_dat[indices[i]].append(obs)


colors = itertools.cycle(["red", "blue", "green", "orange",
                          "yellow", "purple", "cyan", "grey", "white",
                          "palegreen", "black", "fuchsia", "royalblue",
                          "maroon", "pink"])
font = {'family': 'normal',
        'weight': 'normal',
        'size': 22}
matplotlib.rc('font', **font)
# Set figure width to 12 and height to 9
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 18
plt.rcParams["figure.figsize"] = fig_size


cluster_details = []
num = 0
ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=4)
for i, j in enumerate(group_dat):
    lat = []
    lon = []
    ages = []
    err = []
    elev = []
    tc = []
    print(len(j))
    for k, obs in enumerate(j):
        ages.append(obs[2])
        err.append(obs[3])
        lon.append(obs[0])
        lat.append(obs[1])
        elev.append(obs[4])
        tc.append(obs[5])
    ax1.scatter(lon, lat, marker='o', edgecolor='black', alpha=1,
                label='Cluster ' + str(i+1) + ', Age: ' +
                str(round(np.mean(ages), 1)) + ', Unc: ' +
                str(round(np.sqrt(np.std(ages)**2 + np.mean(err)**2), 1)),
                color=next(colors))
    # Create a tuple
    cluster_details.append({"lon": lon,
                            "lat": lat,
                            "age": ages,
                            "unc": round(np.sqrt(np.std(ages)**2 +
                                            np.mean(err)**2), 2),
                            "ele": elev,
                            "nobs": len(lat),
                            "tc": tc })
    num += 1
ax1.legend(loc="lower right", markerscale=1., scatterpoints=1,
           fontsize=10, framealpha=1, borderpad=1)
plt.show()


# Create a csv file with the details of the clusters
for i, clust in enumerate(cluster_details):
    for k in xrange(clust['nobs']):
        print clust['age'][k]

        with open('/home/michaiko/Dropbox/Dropbox/VUW/PhD/'
                'Codes/Modeling_1/Latest_codes/modelling_codes/tc_data/'
                'Thermo_clusters.csv', 'a') as f:
            f.write('{}, {}, {}, {}, {}, {}\n'. format(clust['lon'][k],
                    clust['lat'][k], clust['age'][k], clust['unc'],
                    clust['ele'][k], clust['tc'][k]))
