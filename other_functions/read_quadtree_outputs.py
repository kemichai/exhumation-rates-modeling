"""
- Read quadtree outputs 
- Calculate ..., ..., ...
- Save lists as .npz files (dictionaries)
- One .npz file for each bin

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
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from pylab import *

from functions import read_csv_file
from functions import CoordRotator

# Define parameters for coordinates rotation and transformation
degrees = -54
inProject = Proj(init='epsg:4326')
outProject = Proj(init='epsg:2193')
orig = [0, 0]
aaa = CoordRotator(orig, np.radians(-54))

# Path to files
inpath = '/Volumes/GeoPhysics_05/users-data/michaiko/Matlab/Dep_distr/Outputs/'
boxes_csv_file = (inpath + 'final_boxes.csv')
values_csv_file = (inpath + 'final_values.csv')

# Read parameters from .csv file
params = read_csv_file(boxes_csv_file, 4)
# Read bin corners
X2 = [float(i) for i in params[1]]
X1 = [float(i) for i in params[0]]
Y1 = [float(i) for i in params[2]]
Y2 = [float(i) for i in params[3]]
# Create tuple of the bin corners rotated back and transformed (lat,lon)
bins_quadtree = []
for i, j in enumerate(X1):
    aX, aY = aaa.inverse(X1[i], Y1[i])
    bX, bY = aaa.inverse(X2[i], Y2[i])
    cX, cY = aaa.inverse(X2[i], Y1[i])
    dX, dY = aaa.inverse(X1[i], Y2[i])

    cx, cy = transform(outProject, inProject, cX, cY)
    bx, by = transform(outProject, inProject, bX, bY)
    dx, dy = transform(outProject, inProject, dX, dY)
    ax, ay = transform(outProject, inProject, aX, aY)

    C = [cx, cy, 0]
    B = [bx, by, 0]
    D = [dx, dy, 0]
    A = [ax, ay, 0]

    lats = [ay, by, cy, dy]
    lons = [ax, bx, cx, dx]
    box = np.asarray(zip(lats, lons))
    # Bin's geometry tuple
    bins_quadtree.append({"index": i+1,
                          "latlon": box})

# Match bins with the data within
cluster_det_csv_file = (inpath + 'cluster_details.csv')
cat_csv_file = (inpath + 'cat_box_num.csv')
# Read param values for each box
params_1 = read_csv_file(cat_csv_file, 8)
cluster_names = [int(i) for i in params_1[-1]]
d_unc = [float(i) for i in params_1[-2]]
dep = [float(i) for i in params_1[-3]]
lon = [float(i) for i in params_1[-4]]
lat = [float(i) for i in params_1[-5]]
# For each cluster find the av distance 
# to the plane of the fault
cl_list = []
for i in cluster_names:
    if i not in cl_list:
        cl_list.append(i)
cl_list.sort()

ENCODING = 'utf-8'
# Create lists within lists for each bin....
clusters = [[] for _ in range(len(cl_list))]
for i, val in enumerate(cl_list):
    print('Cluster ' + str(val))
    for k, ev in enumerate(dep):
        if val == cluster_names[k]:
            clusters[i].append({"index": cluster_names[k],
                                "lat": lat[k],
                                "lon": lon[k],
                                "dep": dep[k],
                                "dep_unc": d_unc[k]})

# Create a temporary tuple with the details of the bins
Bin_details_ = []
for k, l in enumerate(clusters):
    number_of_observations = len(clusters[k])
    if number_of_observations > 10:
        Depths = [[] for _ in range(number_of_observations)]
        Depth_uncs = [[] for _ in range(number_of_observations)]
        lats_ = [[] for _ in range(number_of_observations)]
        lons_ = [[] for _ in range(number_of_observations)]
        for i, j in enumerate(clusters[k]):
            Depths[i] = j['dep']
            Depth_uncs[i] = j['dep_unc']/1000
            lats_[i] = j['lat']
            lons_[i] = j['lon']
        for m, n in enumerate(bins_quadtree):
            if j['index'] == n['index']:
                Bin_details_.append({"index": n['index'],
                                     "box_corn": n['latlon'],
                                     "Nobs": number_of_observations,
                                     "Depths": Depths,
                                     "Depth_unc": Depth_uncs,
                                     "lat": lats_,
                                     "lon": lons_})
#####################################################
# Loop through the loop and rename the boxes starting
# from 1 and increasing....
#####################################################
Bin_details = []
for i, j in enumerate(Bin_details_):

    Bin_details.append({"index": j['index'],
                        "box_corn": j['box_corn'],
                        "Nobs": j['Nobs'],
                        "Depths": j['Depths'],
                        "Depth_unc": j['Depth_unc'],
                        "lat": j['lat'],
                        "lon": j['lon']})
#####################################################
# The tuple Bin_details now contains almost all the info needed
#####################################################

# Create two extra lists... X and Y
Y = []
Y_unc = []
for i, j in enumerate(Bin_details):
    # sorting depths and uncertainties at the same time
    a = [list(x) for x in zip(*sorted(zip(j['Depths'], j['Depth_unc']),
                              reverse=True,
                              key=lambda pair: pair[0]))]
    Y.append(a[0])
    Y_unc.append(a[1])
X = []
for m, k in enumerate(Bin_details):
    cumul = [[] for _ in range(k["Nobs"])]

    for i, j in enumerate(range(k["Nobs"])):
        cumul[i] = j * 100/(k["Nobs"]-1)
    X.append(cumul)

# Another temporary tuple...
# NOOOT
Final_box_details = {}
num = 0
for x, y, er in zip(X, Y, Y_unc):
    # Get rid of the upper 90 and 
    # lower 10 percent of 
    # the distributions
    x_ = []
    y_ = []
    er_ = []
    for i, j in enumerate(x):
        if j >= 10 and j <= 90:
            x_.append(j)
            y_.append(y[i])
            er_.append(er[i])
            # print j
    x1 = np.asarray(x_)
    y1 = np.asarray(y_)
    er1 = np.asarray(er_)
    ##############################################################
    # Get the slope of the lines....
    slope, intercept, r_value, p_value, std_err = stats.linregress(x1, y1)
    # values converts it into a numpy array
    X_ = x1.reshape(-1, 1)
    # -1 means that calculate the dimension of rows, but have 1 column
    Y_ = y1.reshape(-1, 1)
    # create object for the class
    linear_regressor = LinearRegression()
    # perform linear regression
    linear_regressor.fit(X_, Y_)
    # make predictions
    Y_pred = linear_regressor.predict(X_)  # make predictions
    reg = LinearRegression().fit(X_, Y_)
    R = reg.score(X_, Y_)    # gives you the R^2 value
    # z0 = reg.intercept_  # equal to reg.intercept_
    z0 = reg.predict(np.array([[0]]))  # equal to reg.intercept_
    z100 = reg.predict(np.array([[100]]))
    ##############################################################
    # New way to get z0 and z100 including uncertatinties on y axis
    def func(x, a, b):
        return  b*x + a
    # curve fit [with only y-error]
    
    popt, pcov = curve_fit(func, x1, y1, sigma=1./(er1*er1))
    perr = np.sqrt(np.diag(pcov))
    # print('fit parameter 1-sigma error')
    # print('———————————–===============')
    # for i in range(len(popt)):
    #     print(str(popt[i])+' +- '+str(perr[i]))

    # prepare confidence level curves
    nstd = 1. # to draw 5-sigma intervals
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr

    fit = func(x1, *popt)
    fit_up = func(x1, *popt_up)
    fit_dw = func(x1, *popt_dw)  
    
    ##############################
    # Fit the function in zero using the 
    # confidence level parameters from all 
    # the data (10-90)
    fit_0 = func(0, *popt)
    fit_up_0 = func(0, *popt_up)
    fit_dw_0 = func(0, *popt_dw)  
    z0_ = fit_0
    z0_unc = fit_up_0 - fit_dw_0
    fit_100 = func(100, *popt)
    fit_up_100 = func(100, *popt_up)
    fit_dw_100 = func(100, *popt_dw)  
    z100_ = fit_100
    z100_unc = fit_up_100 - fit_dw_100
    ####################################################
    # Create the final tuple that has all the values....
    Final_box_details = {"index": Bin_details[num]['index'],
                              "box_corners": Bin_details[num]['box_corn'],
                              "hdep": Bin_details[num]['Depths'],
                              "latit": Bin_details[num]['lat'],
                              "longi": Bin_details[num]['lon'],
                              "h_unc": Bin_details[num]['Depth_unc'],
                              "Predicted_0": z0_,
                              "Predicted_0_unc": z0_unc,
                              "Predicted_100": z100_,
                              "Predicted_100_unc": z100_unc,
                              "Rsquared": R,
                              "Line_slope": slope,
                              "Nobs": Bin_details[num]['Nobs']}
    num += 1
    # Define bin name to be saved
    bin_name = Final_box_details['index']
    dictout = Final_box_details
    np.savez("npz_files/params_bin_" + str(bin_name) +".npz", **dictout)
