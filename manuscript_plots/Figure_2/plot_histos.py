"""Plot the earthquake depth histogram of all the data used as part 
of Chapter 3.

Author: KM
April 2019
"""
from shapely.geometry import Polygon, Point
import csv
import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import matplotlib.pyplot as plt
import os

def read_csv_file(path, num_of_columns, delimitah=','):
    """Return a number of lists (number of columns)."""
    details_file = path
    import csv

    lists = [[] for i in range(num_of_columns)]
    for i in range(num_of_columns):
        print(i)
        c_reader = csv.reader(open(details_file, 'r'), delimiter=delimitah)
        lists[i] = list(zip(*c_reader))[i]
    return lists


###############################################################################
# Read the csv file that contains the vertical uncertainties
working_dir = os.getcwd()
file_name = '/dataset_all.csv'
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
mag = list(zip(*c_reader))[9]
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
dep_unc = list(zip(*c_reader))[11]
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
# Earthquake catalog details
lat = [float(i) for i in lat]
lon = [float(i) for i in lon]
dep = [float(i) for i in dep]
dep_unc = [float(i) for i in dep_unc]
mag = [float(i) for i in mag]


# a = []
# for i, j in enumerate(uplifts):
#         print j['nobs']
#         a.append(j['nobs'])

# dep = []
# for i, j in enumerate(uplifts):
#         if j['ind'] == 47 or j['ind'] == 49 or j['ind']==57 or j['ind']==59:
#                 dep.extend(j['depths'])


# GEONET STUFF
#read the Geonet csv file located in Dropbox/VUW/PhD/preliminary....

csv_file = working_dir + '/GEONET.csv'
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
glon = list(zip(*c_reader))[1]
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
glat = list(zip(*c_reader))[2]
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
gmag = list(zip(*c_reader))[3]
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
gdep = list(zip(*c_reader))[4]
c_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
gdate = list(zip(*c_reader))[0]

glat = [float(i) for i in glat ]
glon = [float(i) for i in glon ]
# date = [float(i) for i in date ]
gmag = [float(i) for i in gmag ]
gdep = [float(i) for i in gdep ]



plats = [ -43.7, -42.6, -43, -44.4]
plons = [ 169.2 , 171.3, 171.8, 169.6]

p= Polygon((np.asarray(list(zip(plats, plons)))))

geonet_depths = np.array([gdep[i] for i in range(len(gmag)) if p.contains(Point(glat[i],glon[i]))])
geonet_mag = np.array([gmag[i] for i in range(len(gmag)) if p.contains(Point(glat[i],glon[i]))])
geonet_depths.tolist()
geomag = geonet_mag.tolist()






###############################################################
# Read thermocron data and 

FT_file = (working_dir + '/Thermochron_data_in_grid.csv')
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
plats = [-44, -42.85, -43.225, -44.375]
plons = [168.95, 171.25, 171.55, 169.25]
p_ = Polygon((np.asarray(list(zip(plats, plons)))))
inside_box = np.array([[X_FT[i], Y_FT[i], Age_FT[i], Err_FT[i],
                        Tc_FT[i], El_FT[i]]
                        for i in range(len(Y_FT))
                        if p_.contains(Point(Y_FT[i], X_FT[i]))])
x_FT = []
y_FT = []
age_FT = []
err_FT = []
tc_FT = []
el_FT = []
for i, j in enumerate(inside_box):
        if j[2] < 200.0:
                # We dont model the cretatious so we dont need those
                x_FT.append(j[0])
                y_FT.append(j[1])
                age_FT.append(j[2])
                error = j[3]
                if error == 0.0:
                        error = 0.1
                        err_FT.append(error)
                        tc_FT.append(j[4])
                        el_FT.append(j[5])       
AFT_age = []
ZFT_age = []
ZHe_age = []
AHe_age = []
for i, j in enumerate(inside_box):
        if j[4] == 100.0:
                AFT_age.append(j[2])
        elif j[4] == 230.0:
                ZFT_age.append(j[2])
        elif j[4] == 70.0:
                AHe_age.append(j[2])
        else:
                ZHe_age.append(j[2])

















################################
# Plotting section of the script
################################
font = {'family': 'normal',
        'weight': 'normal',
        'size': 18}
matplotlib.rc('font', **font)
# Set figure width to 12 and height to 9
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size
################################
################################
bins = np.arange(-1.5, 41.5, 1)
ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
ax1.hist(dep, bins, histtype='step', orientation='horizontal',
         color='black',facecolor='grey', alpha=0.9, linewidth=1.5, 
         edgecolor='k',fill=True, label='This study')
ax1.hist(geonet_depths, bins, histtype='step', orientation='horizontal',
         color='black',facecolor='orange', alpha=0.7, linewidth=1.5, 
         edgecolor='k',fill=True, label='GeoNet')             
ax1.set_ylim([0, 30])
ax1.set_xlim([0, 1000])
ax1.set_ylabel('Depth (km)', fontsize=18)
ax1.set_xlabel(r'Number of events', fontsize=18)
plt.gca().invert_yaxis()
# plt.axhline(np.median(dep), color='k', linestyle='dashed', linewidth=2, label='Mean')
plt.axhline(np.mean(dep), color='k', linestyle='dashed', linewidth=2, 
            label='Mean (' +str(round(np.mean(dep),1)) + ' km)' )
plt.axhline(np.percentile(dep,90), color='k', linestyle='dotted', linewidth=2, 
            label='90th perc (' +str(round(np.percentile(dep,90),1)) + ' km)' )
ax1.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)

# Plot
bins = np.arange(-0.5, 50.5, 1)
ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)
ax2.hist(age_FT, bins, histtype='step', orientation='vertical',
         color='black',facecolor='grey', alpha=0.9, linewidth=1.5, 
         edgecolor='k',fill=True, label='All ages')
# ax1.hist(AFT_age, bins, histtype='step', orientation='vertical',
#          color='blue',facecolor='blue', alpha=0.9, linewidth=.5, 
#          edgecolor='blue',fill=True, label='AFT')
ax2.hist(ZFT_age, bins, histtype='step', orientation='vertical',
         alpha=0.7, linewidth=1.5, linestyle=':',facecolor='green',
         color='green',
         edgecolor='black',fill=True, label='ZFT')  
ax2.hist(AFT_age, bins, histtype='step', orientation='vertical',
         alpha=0.7, linewidth=1.5, linestyle='-',facecolor='red',
         color='red',
         edgecolor='black',fill=True, label='AFT')  
ax2.hist(ZHe_age, bins, histtype='step', orientation='vertical',
         alpha=0.7, linewidth=1.5, facecolor='blue',
         color='blue', 
         edgecolor='black',fill=True, label='ZHe',linestyle='--')  
ax2.hist(AHe_age, bins, histtype='step', orientation='vertical',
         alpha=0.7, linewidth=1.5, facecolor='orange',
         color='orange', 
         edgecolor='black',fill=True, label='AHe',linestyle='-.')          
# ax2.hist(AHe_age, bins, histtype='step', orientation='vertical',
#          alpha=0.7, linewidth=2.5, facecolor='blue',
#          color='blue', 
#          edgecolor='black',fill=True, label='AHe',linestyle='-.')  
# ax1.hist(ZHe_age, bins, histtype='step', orientation='vertical',
#          color='green',facecolor='green', alpha=0.9, linewidth=.5, 
#          edgecolor='green',fill=True, label='ZHe')   
plt.xticks(np.arange(0, 25, 2))
ax2.set_xlim([0, 10])
ax2.set_ylim([0, 50])
# ax1.axvline(2, 0, 90, lw=2, color='k',linestyle='--')
# ax2.axvline(10, 0, 90, lw=2, color='k',linestyle='--')
ax2.set_xlabel('Ages (Ma)', fontsize=18)
ax2.set_ylabel(r'Number of observations', fontsize=18)
ax2.legend(loc="upper right", markerscale=1., scatterpoints=1, fontsize=14)

plt.show()








