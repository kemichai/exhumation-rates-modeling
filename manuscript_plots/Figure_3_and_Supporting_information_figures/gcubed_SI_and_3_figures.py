#!/usr/bin/env python.
"""
Script that reads the quadtree (matlab) codes outputs and creates a number of
plots that are included in the supporting information of the gcubed manuscript
on the exhumation and thermal structure in the central Southern Alps.

1. Cumulative number of eqz vs hypocentral depths for each box (scatter)
2. Cumulative number of eqz vs hypocentral depths for each box (lines fitting
the data for each box with regression least squares and the Z0 and Z100%
predicted values)
3. Scatter plot of Z0 and Z100

A number of text files that can be used with GMT5 to plot the boxes collored
according to some parameters (Z0, Z100, Z0-Z100,...)

=============================================
Requirements:
    * sklearn (conda install -c intel scikit-learn)
    *
=============================================

Te Whare wananga o te upoko o te Ika a Maui
VUW
February 2019
Author: Konstantinos Michailos
"""

from matplotlib.ticker import PercentFormatter
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from scipy import stats
import os
from main_functions.functions import *
from pyproj import Proj, transform
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

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


# Define working directory
working_dir = os.getcwd()
# Read outputs
path = '/quadtree_outputs/'

boxes_csv_file = (working_dir + path + 'final_boxes.csv')
values_csv_file = (working_dir + path + 'final_values.csv')

c_reader = csv.reader(open(boxes_csv_file, 'r'), delimiter=',')
X2 = list(zip(*c_reader))[1]
c_reader = csv.reader(open(boxes_csv_file, 'r'), delimiter=',')
X1 = list(zip(*c_reader))[0]
c_reader = csv.reader(open(boxes_csv_file, 'r'), delimiter=',')
Y1 = list(zip(*c_reader))[2]
c_reader = csv.reader(open(boxes_csv_file, 'r'), delimiter=',')
Y2 = list(zip(*c_reader))[3]

# These are the four edges of each box defined by quadtree
X1 = np.asarray([float(i) for i in X1])
X2 = np.asarray([float(i) for i in X2])
Y1 = np.asarray([float(i) for i in Y1])
Y2 = np.asarray([float(i) for i in Y2])

# These are the values within each box
c_reader = csv.reader(open(values_csv_file, 'r'), delimiter=',')
avdep = list(zip(*c_reader))[0]
avdep = np.asarray([float(i) for i in avdep])

# Define the projections
degrees = -54
inProject = Proj(init='epsg:4326')
outProject = Proj(init='epsg:2193')
inProj = inProject
outProj = outProject
orig = [0, 0]
aaa = CoordRotator(orig, np.radians(-54))

bins_quadtree = []
for i, j in enumerate(X1):
    aX, aY = aaa.inverse(X1[i], Y1[i])
    bX, bY = aaa.inverse(X2[i], Y2[i])
    cX, cY = aaa.inverse(X2[i], Y1[i])
    dX, dY = aaa.inverse(X1[i], Y2[i])

    cx, cy = transform(outProj, inProj, cX, cY)
    bx, by = transform(outProj, inProj, bX, bY)
    dx, dy = transform(outProj, inProj, dX, dY)
    ax, ay = transform(outProj, inProj, aX, aY)

    C = [cx, cy, 0]
    B = [bx, by, 0]
    D = [dx, dy, 0]
    A = [ax, ay, 0]

    lats = [ay, by, cy, dy]
    lons = [ax, bx, cx, dx]
    box = np.asarray(zip(lats, lons))
#############################################################################
    bins_quadtree.append({"index": i+1,
                          "latlon": box})
#############################################################################


cluster_file = (working_dir + path + 'cluster_details.csv')
cat_file = (working_dir + path + 'cat_box_num.csv')

cat_csv_file = (cat_file)
cluster_det_csv_file = (cluster_file)

c_reader = csv.reader(open(cat_csv_file, 'r'), delimiter=',')
cluster_names = list(zip(*c_reader))[-1]
c_reader = csv.reader(open(cat_csv_file, 'r'), delimiter=',')
d_unc = list(zip(*c_reader))[-2]
c_reader = csv.reader(open(cat_csv_file, 'r'), delimiter=',')
dep = list(zip(*c_reader))[-3]
c_reader = csv.reader(open(cat_csv_file, 'r'), delimiter=',')
lon = list(zip(*c_reader))[-4]
c_reader = csv.reader(open(cat_csv_file, 'r'), delimiter=',')
lat = list(zip(*c_reader))[-5]

lat = [float(i) for i in lat]
dep = [float(i) for i in dep]
lon = [float(i) for i in lon]
d_unc = [float(i) for i in d_unc]
cluster_names = [int(i) for i in cluster_names]

# For each cluster find the av distance to the plane of the fault
cl_list = []
for i in cluster_names:
    if i not in cl_list:
        cl_list.append(i)
cl_list.sort()

ENCODING = 'utf-8'
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

Bin_details_ = []
for k, l in enumerate(clusters):
    number_of_observations = len(clusters[k])
    if number_of_observations > 10:
        print('Number of eqz in each bin: ' + str(number_of_observations))

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

# Tuple Bin_details now contains almost all the info needed
# We need to make a new one with the predicted values.
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

###############################
# Plotting section of the script
###############################

###############################################################################
# Single plot for each box
###############################################################################
# Plot the first chunk of the data
X_a = []
Y_a = []
Y_unc_a = []
box_num_a = []
box_nobs_a = []
for i, j in enumerate(X):
    # First 30 observations
    if i < 30:
        X_a.append(j)
        Y_a.append(Y[i])
        Y_unc_a.append(Y_unc[i])
        box_num_a.append(Bin_details[i]['index'])
        box_nobs_a.append(Bin_details[i]['Nobs'])
###############################################################################
font = {'family': 'normal',
        'weight': 'normal',
        'size': 10}
matplotlib.rc('font', **font)
# Set figure width to 12 and height to 9
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 18
fig_size[1] = 18
plt.rcParams["figure.figsize"] = fig_size
###############################################################################
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)
num = 0
for i, j in enumerate(range(0, len(X_a))):
    print(i)
    ax1 = fig.add_subplot(6, 5, i+1)
    ax1.errorbar(X_a[i], Y_a[i], yerr=Y_unc_a[i], fmt='o',
                 capsize=0.1, elinewidth=.1, markeredgewidth=.1,
                 label='Box: ' + str(box_num_a[i]) +
                 ', Nobs: ' + str(box_nobs_a[i]), alpha=0.5)
    num += 1
    ax1.legend(bbox_to_anchor=(.12, 0.25), loc=2,
               borderaxespad=0., fontsize=10)
    # ax1.set_ylabel('Hypocentral Depth (km)', fontsize=20)
    # ax1.set_xlabel('Cumulative %', fontsize=20)
    ax1.xaxis.set_label_position('top')
    ax1.set_ylim([0, 30])
    ax1.set_xlim([0, 100])
    ax1.vlines(10, 0, 30, lw=0.5, color='k', linestyle='-')
    ax1.vlines(90, 0, 30, lw=0.5, color='k', linestyle='-')
    plt.xticks(np.arange(0, 125, 25))
    plt.yticks(np.arange(0, 35, 5))
    ax1.xaxis.tick_top()
    plt.gca().invert_yaxis()
    fig_name = working_dir + '/S1_a.png'
    plt.savefig(fig_name, bbox_inches="tight", format='png')
plt.show()

# Plot the next chunk of data
X_b = []
Y_b = []
Y_unc_b = []
box_num_b = []
box_nobs_b = []
R_b = []
for i, j in enumerate(X):
    if i >= 30:
        X_b.append(j)
        Y_b.append(Y[i])
        Y_unc_b.append(Y_unc[i])
        box_num_b.append(Bin_details[i]['index'])
        box_nobs_b.append(Bin_details[i]['Nobs'])

fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)
for i, j in enumerate(range(0, len(X_b))):
    print(i)
    ax1 = fig.add_subplot(7, 5, i+1)
    ax1.errorbar(X_b[i], Y_b[i], yerr=Y_unc_b[i], fmt='o',
                 capsize=0.1, elinewidth=.1, markeredgewidth=.1,
                 label='Box: ' + str(box_num_b[i]) +
                 ', n: ' + str(box_nobs_b[i]), alpha=0.5)
    ax1.legend(bbox_to_anchor=(.12, 0.25),
               loc=2, borderaxespad=0., fontsize=10)
    # ax1.set_ylabel('Hypocentral Depth (km)', fontsize=20)
    # ax1.set_xlabel('Cumulative %', fontsize=20)
    ax1.xaxis.set_label_position('top')
    ax1.set_ylim([0, 30])
    ax1.set_xlim([0, 100])
    ax1.vlines(10, 0, 30, lw=0.5, color='k', linestyle='-')
    ax1.vlines(90, 0, 30, lw=0.5, color='k', linestyle='-')
    plt.xticks(np.arange(0, 125, 25))
    plt.yticks(np.arange(0, 35, 5))
    ax1.xaxis.tick_top()
    plt.gca().invert_yaxis()
    fig_name = working_dir + '/S1_b.png'
    plt.savefig(fig_name, bbox_inches="tight", format='png')
plt.show()
###############################################################################
# Make the same plots but histograms..
###############################################################################
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)
bins = np.arange(-0.5, 30.5, 2)
for i, j in enumerate(range(0, len(X_a))):
    print(i)
    ax1 = fig.add_subplot(6, 5, i+1)
    ax1.hist(Y_a[i], weights=np.ones(len(Y_a[i])) / len(Y_a[i]),
             histtype='step', orientation='horizontal',
             color='black', facecolor='grey', linewidth=1.5,
             edgecolor='k', label='Box: ' + str(box_num_a[i]) +
             ', n: ' + str(box_nobs_a[i]), alpha=0.9, fill=True)
    plt.axhline(np.mean(Y_a[i]), color='k', linestyle='dashed', linewidth=1)
    plt.axhline(np.percentile(Y_a[i], 90), color='k', linestyle='dotted',
                linewidth=1)
    ax1.legend(bbox_to_anchor=(.12, 0.25), loc=2,
               borderaxespad=0., fontsize=10)
    # ax1.set_ylabel('Hypocentral Depth (km)', fontsize=20)
    # ax1.set_xlabel('Cumulative %', fontsize=20)
    ax1.xaxis.set_label_position('top')
    # if box_nobs_a[i]>100:
    #     ax1.set_xlim([0, 100])
    # else:
    #     ax1.set_xlim([0, 50])
    # ax1.set_xlim([0, 80])
    ax1.set_ylim([0, 30])
    # ax1.set_xlim([0, 80])
    # plt.xticks(np.arange(0, 125, 25))
    plt.yticks(np.arange(0, 35, 5))
    ax1.xaxis.tick_top()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
    fig_name = working_dir + '/S2_a.png'
    plt.savefig(fig_name, bbox_inches="tight", format='png')
plt.show()


fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)
bins = np.arange(-0.5, 30.5, 2)
for i, j in enumerate(range(0, len(X_b))):
    print(i)
    ax1 = fig.add_subplot(7, 5, i+1)
    ax1.hist(Y_b[i], weights=np.ones(len(Y_b[i])) / len(Y_b[i]),
             histtype='step', orientation='horizontal',
             color='black', facecolor='grey', linewidth=1.,
             edgecolor='k', label='Box: ' + str(box_num_b[i]) +
             ', n: ' + str(box_nobs_b[i]), alpha=0.9, fill=True)
    ax1.legend(bbox_to_anchor=(.12, 0.25), loc=2,
               borderaxespad=0., fontsize=10)
    # ax1.set_ylabel('Hypocentral Depth (km)', fontsize=20)
    # ax1.set_xlabel('Cumulative %', fontsize=20)
    ax1.xaxis.set_label_position('top')
    ax1.set_ylim([0, 30])
    # ax1.set_xlim([0, 80])
    # plt.xticks(np.arange(0, 125, 25))
    plt.yticks(np.arange(0, 35, 5))
    ax1.xaxis.tick_top()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
    fig_name = working_dir + '/S2_b.png'
    plt.savefig(fig_name, bbox_inches="tight", format='png')
plt.show()


###############################################################################
###############################################################################
num = 0
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)
# ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=4)
for x, y, er in zip(X_a, Y_a, Y_unc_a):
    x_ = []
    y_ = []
    for i, j in enumerate(x):
        if j >= 10 and j <= 90:
            x_.append(j)
            y_.append(y[i])
            print(j)
    x1 = np.asarray(x_)
    y1 = np.asarray(y_)
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
    z0 = reg.predict(np.array([[0]]))  # equal to reg.intercept_
    z100 = reg.predict(np.array([[100]]))

    ###########################################################################
    # Create the final tuple that has all the values....
    ax1 = fig.add_subplot(6, 5, num+1)
    ##################################
    x_2 = reg.score(X_, Y_)
    ax1.scatter(x, y, color='grey',
                label='Box: ' + str(box_num_a[num]) +
                ', Nobs: ' + str(box_nobs_a[num]) +
                ', R: ' + str(round(R, 2)), alpha=0.5)
    # ax1.plot(X_, Y_pred, color='red', alpha=0.5, label='Zo: ' +
    #          str(z0[0][0])+', Z100: ' + str(z100[0][0]) +
    #          ', $R^2$: ' + str(R))
    ax1.plot(X_, Y_pred, color='red', alpha=0.5)
    ax1.scatter(0.5, z0, marker='s', color='red')
    ax1.scatter(99.5, z100, marker='s', color='red')
    ax1.legend(bbox_to_anchor=(-.08, 0.25), loc=2,
               borderaxespad=0., fontsize=10, markerscale=0, frameon=False)
    num += 1
    ax1.xaxis.set_label_position('top')
    ax1.set_ylim([0, 30])
    ax1.set_xlim([0, 100])
    ax1.vlines(10, 0, 30, lw=0.5, color='k', linestyle='-')
    ax1.vlines(90, 0, 30, lw=0.5, color='k', linestyle='-')
    plt.xticks(np.arange(0, 125, 25))
    plt.yticks(np.arange(0, 35, 5))
    ax1.xaxis.tick_top()
    plt.gca().invert_yaxis()
    fig_name = working_dir + '/S3_a.png'
    plt.savefig(fig_name, bbox_inches="tight", format='png')
plt.show()


###############################################################################
num = 0
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)
# ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=4)
for x, y, er in zip(X_b, Y_b, Y_unc_b):
    # ax1.plot(x, y, "k--", lw=1.)
    # ax1.errorbar(x, y, yerr=er, fmt='o',
    #              capsize=0.5, elinewidth=.5, markeredgewidth=.5,
    #              label=str(N_obs[num]), alpha=0.5)
    x_ = []
    y_ = []
    for i, j in enumerate(x):
        if j >= 10 and j <= 90:
            x_.append(j)
            y_.append(y[i])
            print(j)
    x1 = np.asarray(x_)
    y1 = np.asarray(y_)

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
    z0 = reg.predict(np.array([[0]]))  # equal to reg.intercept_
    z100 = reg.predict(np.array([[100]]))

    ###########################################################################
    ax1 = fig.add_subplot(7, 5, num+1)
    ##################################
    x_2 = reg.score(X_, Y_)
    ax1.scatter(x, y, color='grey',
                label='Box: ' + str(box_num_b[num]) +
                ', Nobs: ' + str(box_nobs_b[num]) +
                ', R: ' + str(round(R, 2)), alpha=0.5)
    # ax1.plot(X_, Y_pred, color='red', alpha=0.5, label='Zo: ' +
    #          str(z0[0][0])+', Z100: ' + str(z100[0][0]) + ', $R^2$: '
    #          + str(R))
    ax1.plot(X_, Y_pred, color='red', alpha=0.5)
    ax1.scatter(0.5, z0, marker='s', color='red')
    ax1.scatter(99.5, z100, marker='s', color='red')
    ax1.legend(bbox_to_anchor=(-.08, 0.25), loc=2,
               borderaxespad=0., fontsize=10, markerscale=0, frameon=False)

    num += 1
    ax1.xaxis.set_label_position('top')
    ax1.set_ylim([0, 30])
    ax1.set_xlim([0, 100])
    ax1.vlines(10, 0, 30, lw=0.5, color='k', linestyle='-')
    ax1.vlines(90, 0, 30, lw=0.5, color='k', linestyle='-')
    plt.xticks(np.arange(0, 125, 25))
    plt.yticks(np.arange(0, 35, 5))
    ax1.xaxis.tick_top()
    plt.gca().invert_yaxis()
    fig_name = working_dir + '/S3_b.png'
    plt.savefig(fig_name, bbox_inches="tight", format='png')
plt.show()


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

X_ex = []
Y_ex = []
Y_unc_ex = []
box_num_ex = []
box_nobs_ex = []
for i, j in enumerate(X):
    # First 30 observations
    if Bin_details[i]['index'] == 17 or Bin_details[i]['index'] == 64 or\
               Bin_details[i]['index'] == 41 or Bin_details[i]['index'] == 74:
        X_ex.append(j)
        Y_ex.append(Y[i])
        Y_unc_ex.append(Y_unc[i])
        box_num_ex.append(Bin_details[i]['index'])
        box_nobs_ex.append(Bin_details[i]['Nobs'])

# 4 of the boxes to add as an example
X_pred_ = []
Y_pred_ = []
Y_err_ = []
z_up = []
z_up_unc = []
z_down = []
z_down_unc = []
num = 0
# ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=4)
for x, y, er in zip(X_ex, Y_ex, Y_unc_ex):
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

    def func(x, a, b):
        return b*x + a
    # curve fit [with only y-error]

    popt, pcov = curve_fit(func, x1, y1, sigma=1./(er1*er1))
    perr = np.sqrt(np.diag(pcov))
    print('fit parameter 1-sigma error')
    print('———————————–===============')
    for i in range(len(popt)):
        print(str(popt[i])+' +- '+str(perr[i]))

    # prepare confidence level curves
    nstd = 1. # to draw 5-sigma intervals
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr

    fit = func(x1, *popt)
    fit_up = func(x1, *popt_up)
    fit_dw = func(x1, *popt_dw)

    # fig, ax = plt.subplots(1)
    # rcParams['xtick.labelsize'] = 18
    # rcParams['ytick.labelsize'] = 18
    # rcParams['font.size']= 20

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

    X_pred_.append(x1)
    Y_pred_.append(fit)
    Y_err_.append(er1)
    z_up.append(z100_)
    z_up_unc.append(z100_unc)
    z_down.append(z0_)
    z_down_unc.append(z0_unc)
    ###########################################################################
    # Create the final tuple that has all the values....
    # ax1 = fig.add_subplot(2, 2, num+1)
    # ##################################

# Major ticks every 20, minor ticks every 10
major_ticks = np.arange(0, 120, 20)
minor_ticks = np.arange(0, 100, 10)
################################
font = {'family': 'normal',
        'weight': 'normal',
        'size': 14}
matplotlib.rc('font', **font)
# Set figure width to 12 and height to 9
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size
################################

ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
ax1.scatter(X_ex[0], Y_ex[0], color='grey', alpha=0.5, label='Box: '
            + str(box_num_ex[0]) + ', n: ' + str(box_nobs_ex[0]))

ax1.errorbar(99, z_up[0], yerr=z_up_unc[0], fmt='ks', ecolor='k',
             markersize=4, capsize=3, elinewidth=1.5, markeredgewidth=.5,
             alpha=1, label='Upper cut-off')
ax1.errorbar(1, z_down[0], yerr=z_down_unc[0], fmt='rs', ecolor='k',
             markersize=4, capsize=3, elinewidth=1.5, markeredgewidth=.5,
             alpha=1, label='Lower cut-off')
ax1.plot(X_pred_[0], Y_pred_[0], color='red', alpha=0.5)
ax1.xaxis.set_label_position('top')
ax1.set_ylim([0, 30])
ax1.set_xlim([0, 100])
ax1.vlines(10, 0, 30, lw=0.5, color='k', linestyle='--')
ax1.vlines(90, 0, 30, lw=0.5, color='k', linestyle='--')
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 35, 5))
ax1.set_xticks(major_ticks)
ax1.set_xticks(minor_ticks, minor=True)
ax1.xaxis.tick_top()
plt.gca().invert_yaxis()
ax1.set_ylabel('Hypocentral Depth (km)', fontsize=16)
ax1.set_xlabel('Cumulative %', fontsize=16)
ax1.legend(loc="lower center", markerscale=1., scatterpoints=1, fontsize=12)
ax1.tick_params(top=True, right=True, left=True, bottom=True)

ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
ax2.scatter(X_ex[1], Y_ex[1], color='grey', alpha=0.5, label='Box: '
            + str(box_num_ex[1]) + ', n: ' + str(box_nobs_ex[1]))
ax2.errorbar(99, z_up[1], yerr=z_up_unc[1], fmt='ks', ecolor='k',
             markersize=4, capsize=3, elinewidth=1.5, markeredgewidth=.5,
             alpha=1)
ax2.errorbar(1, z_down[1], yerr=z_down_unc[1], fmt='rs', ecolor='k',
             markersize=4, capsize=3, elinewidth=1.5,
             markeredgewidth=.5, alpha=1)
ax2.plot(X_pred_[1], Y_pred_[1], color='red', alpha=0.5)
ax2.xaxis.set_label_position('top')
ax2.set_ylim([0, 30])
ax2.set_xlim([0, 100])
ax2.vlines(10, 0, 30, lw=0.5, color='k', linestyle='--')
ax2.vlines(90, 0, 30, lw=0.5, color='k', linestyle='--')
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 35, 5))
ax2.set_xticks(major_ticks)
ax2.set_xticks(minor_ticks, minor=True)
plt.gca().invert_yaxis()
ax2.tick_params(top=True, right=True, left=True, bottom=True)
ax2.xaxis.set_label_position('bottom')
ax2.legend(loc="lower center", markerscale=1., scatterpoints=1, fontsize=12)

ax3 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
ax3.scatter(X_ex[2], Y_ex[2], color='grey', alpha=0.5, label='Box: '
            + str(box_num_ex[2]) + ', n: ' + str(box_nobs_ex[2]))

ax3.errorbar(99, z_up[2], yerr=z_up_unc[2], fmt='ks', ecolor='k',
             markersize=4, capsize=3, elinewidth=1.5,
             markeredgewidth=.5, alpha=1)
ax3.errorbar(1, z_down[2], yerr=z_down_unc[2], fmt='rs', ecolor='k',
             markersize=4, capsize=3, elinewidth=1.5,
             markeredgewidth=.5, alpha=1)
ax3.plot(X_pred_[2], Y_pred_[2], color='red', alpha=0.5)
ax3.xaxis.set_label_position('top')
ax3.set_ylim([0, 30])
ax3.set_xlim([0, 100])
ax3.vlines(10, 0, 30, lw=0.5, color='k', linestyle='--')
ax3.vlines(90, 0, 30, lw=0.5, color='k', linestyle='--')
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 35, 5))
ax3.set_xticks(major_ticks)
ax3.set_xticks(minor_ticks, minor=True)
ax3.xaxis.tick_top()
plt.gca().invert_yaxis()
ax3.tick_params(top=True, right=True, left=True, bottom=True)
ax3.legend(loc="lower center", markerscale=1., scatterpoints=1, fontsize=12)

ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
ax4.scatter(X_ex[3], Y_ex[3], color='grey', alpha=0.5, label='Box: '
            + str(box_num_ex[3]) + ', n: ' + str(box_nobs_ex[3]))
ax4.errorbar(99, z_up[3], yerr=z_up_unc[3], fmt='ks', ecolor='k',
             markersize=4, capsize=3, elinewidth=1.5,
             markeredgewidth=.5, alpha=1)
ax4.errorbar(1, z_down[3], yerr=z_down_unc[3], fmt='rs', ecolor='k',
             markersize=4, capsize=3, elinewidth=1.5,
             markeredgewidth=.5, alpha=1)
ax4.plot(X_pred_[3], Y_pred_[3], color='red', alpha=0.5)
ax4.xaxis.set_label_position('bottom')
ax4.yaxis.tick_right()
ax4.tick_params(top=True, right=True, left=True, bottom=True)
ax4.set_ylim([0, 30])
ax4.set_xlim([0, 100])
ax4.vlines(10, 0, 30, lw=0.5, color='k', linestyle='--')
ax4.vlines(90, 0, 30, lw=0.5, color='k', linestyle='--')
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 35, 5))
ax4.set_xticks(major_ticks)
ax4.set_xticks(minor_ticks, minor=True)
plt.gca().invert_yaxis()
ax4.legend(loc="lower center", markerscale=1., scatterpoints=1, fontsize=12)
fig_name = working_dir + '/gcubed_figure3.png'
plt.savefig(fig_name, bbox_inches="tight", format='png')
plt.show()
