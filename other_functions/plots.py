"""
Script that makes a number of plots.

Attention: the plots below will only work after running the codes in
           Seism_and_thermo.py

"""
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")

working_dir = "/home/kostas/Desktop/Codes_and_stuff/bitbucket/thermal_exh_gcubed/exh_rates_sa/"
file_name = 'mod_uplifts_alpha1.txt'
with open(working_dir + 'Output_files_alpha1/' + file_name, 'r') as f:
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
working_dir = "/home/kostas/Desktop/Codes_and_stuff/bitbucket/thermal_exh_gcubed/exh_rates_sa/"
file_name = 'dataset_all.csv'
eq_cat_file = working_dir + file_name
# Define paramters to be used in the calculations
params = read_csv_file(eq_cat_file, 12)
lon = [float(i) for i in params[7]]
lat = [float(i) for i in params[6]]
dep = [float(i) for i in params[8]]

earthquake_temp_a = []
earthquake_dep_a = []
for i, j in enumerate(lon):
    exh = interpolated_surf_a(lon[i], lat[i])
    eq_t = eq_temp(exh[0], dep[i])
    earthquake_temp_a.append(eq_t)
    earthquake_dep_a.append(dep[i])


# #########################################################
# Calculate temperature of each earthquake and plot a histo

# Use outputs from running the minimization process and define an interpolated
# surface for the exhumation rates
interpolated_surf = interp_surf(final_uplifts[1], final_uplifts[0],
                                    final_uplifts[2])
# Read the individual earthquake locations and in particular
# the csv file that contains the vertical uncertainties
working_dir = "/home/kmichall/Desktop/Codes/bitbucket/exhumation_rates_southern_alps/dataset/"
file_name = 'dataset_all.csv'
eq_cat_file = working_dir + file_name
# Define paramters to be used in the calculations
params = read_csv_file(eq_cat_file, 12)
lon = [float(i) for i in params[7]]
lat = [float(i) for i in params[6]]
dep = [float(i) for i in params[8]]

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

earthquake_temp = []
earthquake_dep = []
for i, j in enumerate(lon):
    exh = interpolated_surf(lon[i], lat[i])
    eq_t = eq_temp(exh[0], dep[i])
    earthquake_temp.append(eq_t)
    earthquake_dep.append(dep[i])

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
         edgecolor='red', fill=True, linestyle='--',label='Steady-state; a=1')
ax1.set_xlim([0, 650])
ax1.set_ylim([0, 500])
ax1.set_xlabel(u'Temperature (℃)', fontsize=18)
ax1.set_ylabel(r'Number of events', fontsize=18)
# plt.gca().invert_yaxis()
# plt.axvline(np.median(earthquake_temp), color='k', linestyle='dashed',
#             linewidth=2,
#             label='Median (' + str(round(np.median(earthquake_temp), 1)) + u' ℃)' )
# plt.axvline(np.percentile(earthquake_temp,10), color='k', linestyle='dotted', linewidth=2,
#              label='10th perc (' +str(round(np.percentile(earthquake_temp,10),1)) + ' C$^\circ$)')
# plt.axvline(np.percentile(earthquake_temp, 90), color='k', linestyle='dotted', linewidth=2,
#             label='90th perc (' + str(round(np.percentile(earthquake_temp, 90), 1)) + u' ℃)' )
# ax1.axvline(res.x[-1], 0, 90, lw=2, color='r',linestyle='-',
#             label=r'T$_{BDT}$ (' + str(round(res.x[-1],1)) + u' ℃)')
plt.axvline(447.3, color='red', linestyle='--', linewidth=2,
            label=r'T$_{BDT}$ (' + str(447.3) + u' ℃)' )
ax1.axvline(467.7, 0, 90, lw=2, color='k',linestyle='-',
            label=r'T$_{BDT}$ (' + str(467.7) + u' ℃)')
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
plt.show()



















#######################################################
# Plot temperature vs hypocentral depths of earthquakes
ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=2)
ax1.scatter(earthquake_temp, earthquake_dep, color='orange',
            edgecolor='black', alpha=0.5)
ax1.set_ylabel(r'Hypocentral depth (km)', fontsize=20)
ax1.set_xlabel(r'Earthquake temperature (C$^\circ$)', fontsize=20)
ax1.set_ylim([0, 35])
ax1.set_xlim([0, 650])
plt.gca().invert_yaxis()
plt.show()














#############################################################
# This is not finished and needs to be checked
# It plots the elevation of the thermochron data vs the
# uplift residuals

# working_dir = "/home/michaiko/Dropbox/Dropbox/VUW/PhD/Codes/Modeling_1/Latest_codes/modelling_codes/"
# file_name = 'delta_vel.csv'
# _file = working_dir + file_name

# params = read_csv_file(_file, 5, delimitah=' ')
# d_v = [float(i) for i in params[3]]
# elev = [float(i) for i in params[4]]

# font = {'family': 'normal',
#         'weight': 'normal',
#         'size': 18}
# matplotlib.rc('font', **font)
# # Set figure width to 12 and height to 9
# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 9
# fig_size[1] = 7
# plt.rcParams["figure.figsize"] = fig_size
# # Plot of tem vs dep of earthquakes
# ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=2)
# ax1.scatter(d_v, elev, color='orange', edgecolor='black', alpha=0.5)
# ax1.set_ylabel(r'Elevation of sample (m)', fontsize=20)
# ax1.set_xlabel(r'Uplift rate residual (mm/yr)', fontsize=20)
# ax1.set_ylim([0, 3500])
# ax1.set_xlim([-30, 30])
# plt.show()


###############################################################
# Read thermocron data and
Tc_path = ('/home/michaiko/Dropbox/Dropbox/VUW/PhD/Codes/'
           'Modeling_1/Latest_codes/modelling_codes/tc_data/')
FT_file = (Tc_path + 'Thermochron_data.csv')
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
for i, j in enumerate(inside_box):
        if j[4] == 100.0:
                AFT_age.append(j[2])
        elif j[4] == 230.0:
                ZFT_age.append(j[2])
        else:
                ZHe_age.append(j[2])
# Plot
bins = np.arange(-0.5, 50.5, 1)
ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
ax1.hist(age_FT, bins, histtype='step', orientation='vertical',
         color='black',facecolor='grey', alpha=0.9, linewidth=1.5,
         edgecolor='k',fill=True)
# ax1.hist(AFT_age, bins, histtype='step', orientation='vertical',
#          color='blue',facecolor='blue', alpha=0.9, linewidth=.5,
#          edgecolor='blue',fill=True, label='AFT')
# ax1.hist(ZFT_age, bins, histtype='step', orientation='vertical',
#          color='red',facecolor='red', alpha=0.9, linewidth=.5,
#          edgecolor='red',fill=True, label='ZFT')
# ax1.hist(ZHe_age, bins, histtype='step', orientation='vertical',
#          color='green',facecolor='green', alpha=0.9, linewidth=.5,
#          edgecolor='green',fill=True, label='ZHe')
plt.xticks(np.arange(0, 25, 5))
ax1.set_xlim([0, 20])
ax1.set_ylim([0, 50])
# ax1.axvline(2, 0, 90, lw=2, color='k',linestyle='--')
ax1.axvline(10, 0, 90, lw=2, color='k',linestyle='--')
ax1.set_xlabel('Ages (Ma)', fontsize=18)
ax1.set_ylabel(r'Number of observations', fontsize=18)
ax1.legend(loc="upper right", markerscale=1., scatterpoints=1, fontsize=14)
plt.show()


################################
# Geothermal gradient values ...
# T0 = temperature at zero depth (sea level)
# T_500 = temperature at 500 m
# Heatflow = (T_500 - T0)*2 (C/km)

T_bdt = res.x[-1]
seis_lon =[]
seis_lat = []
geo_grad = []
for i, j in enumerate(seis_obs):
        seis_lon.append(j[0])
        seis_lat.append(j[1])
        depth = j[2] *(-1) /1000


        gradient = (T_bdt - 13) / depth
        print(depth, gradient)
        geo_grad.append(gradient)

# earthquake_temp
# earthquake_dep
geo_grad_in_eq_loc = []
for i, j in enumerate(earthquake_temp):
    gradient = (earthquake_temp[i] - 13.) / earthquake_dep[i]
    geo_grad_in_eq_loc.append(gradient)
    print(earthquake_dep[i],gradient)
# This here calculates the geothermal gradients for all the
# estimated exhumation rates
gradients = []
for exh in res.x[0:-3]:
    T500 = eq_temp(exh, 0.5)
    gradient = (T500 - 13.)*2  # C/km
    gradients.append(gradient)
    print(exh, T500, gradient)










# TODO: Turn this into a function
interpolated_surf = interp_surf(final_uplifts[1], final_uplifts[0],
                                final_uplifts[2])

start_lon_ = 169.13
start_lat = -43.87
end_lon = 171.13
end_lat = -42.87

lon_range_SW = np.arange(169.13, 169.53, 0.2)
lat_range_SW = np.arange(-43.87, -44.27, -0.1)

lon_range_NE = np.arange(171.13, 171.53, 0.2)
lat_range_NE = np.arange(-42.87, -43.27, -0.1)


x = []
y = []
for i, j in enumerate(lon_range_SW):
    start_lon = j
    start_lat = lat_range_SW[i]
    print(start_lat, start_lon)
    # end_lon = lon_range_NE[i]
    # end_lat = lat_range_NE[i]
    # x.append(start_lon)
    # y.append(start_lat)

    for k in range(12):
        lon = start_lon + 0.2 * k
        lat = start_lat + 0.1 * k
        print(lon,lat)
        x.append(lon)
        y.append(lat)

plt.scatter(x,y)
plt.show()

z0 = 0.
year = 365.25*24*60*60
z1 = p[-3]
z = np.linspace(z0, z1, 100)

a = []
for i, j in enumerate(x):
    pred_exh_per = interpolated_surf(x[i], y[i])
    a.append(pred_exh_per)

    temp_prof_par = temperature_profile(z, pred_exh_per[0]/(1e+3 * year), p)
    z_100 = predict_depth(temp_prof_par, z, 100)/1000*(-1)

    with open('Exhum_grid.txt', 'a') as of:
        of.write('{}, {}, {}\n'.format(round(x[i], 3), round(y[i], 3), pred_exh_per[0]))




# TODO: Sort this mess out!!!
# NOTE: The lines below define the locations of the grid points of the model.
#       And calculate exhum and the depths of 100, 200 etc centigrades on each one of these points.white
#       NEED to run the run_models with ipython first to run the code below
first_point = [169.13, -43.87]
second_point = [169.23, -43.995]
third_point = [169.33, -44.12]
fourth_point = [169.43, -44.245]
fivth_point = [169.53, -44.37]
# First line of points runs along the AF on the footwall
# this line has fixed uplift rates that are close to zero
lon = []
lat = []
vel = []
lon1 = 169.13
lat1 = -43.87
for i in range(10):
    lon1 = lon1 + 0.2
    lon.append(lon1)
    lat1 = lat1 + 0.1
    lat.append(lat1)
    vel.append(0.1)
lon.append(first_point[0])
lat.append(first_point[1])
vel.append(0.1)

# Second line of points parallel to the first one (further southeast) runs
# in the hanging wall side and roughly goes through Aoraki Mount Cook
# Uplift rate values here are parameters that will be readjusted during the
# optimize.minimize step
lon.append(second_point[0])
lat.append(second_point[1])
vel.append(vel_A1)
lon.append(second_point[0] + 0.2)
lat.append(second_point[1] + 0.1)
vel.append(vel_A2)
lon.append(second_point[0] + 0.4)
lat.append(second_point[1] + 0.2)
vel.append(vel_A3)
lon.append(second_point[0] + 0.6)
lat.append(second_point[1] + 0.3)
vel.append(vel_A4)
lon.append(second_point[0] + 0.8)
lat.append(second_point[1] + 0.4)
vel.append(vel_A5)
lon.append(second_point[0] + 1.0)
lat.append(second_point[1] + 0.5)
vel.append(vel_A6)
lon.append(second_point[0] + 1.2)
lat.append(second_point[1] + 0.6)
vel.append(vel_A7)
lon.append(second_point[0] + 1.4)
lat.append(second_point[1] + 0.7)
vel.append(vel_A8)
lon.append(second_point[0] + 1.6)
lat.append(second_point[1] + 0.8)
vel.append(vel_A9)
lon.append(second_point[0] + 1.8)
lat.append(second_point[1] + 0.9)
vel.append(vel_A10)
lon.append(second_point[0] + 2.0)
lat.append(second_point[1] + 1.0)
vel.append(vel_A11)
# Third line of points parallel to the second one and further southeast
lon.append(third_point[0])
lat.append(third_point[1])
vel.append(vel_B1)
lon.append(third_point[0] + 0.2)
lat.append(third_point[1] + 0.1)
vel.append(vel_B2)
lon.append(third_point[0] + 0.4)
lat.append(third_point[1] + 0.2)
vel.append(vel_B3)
lon.append(third_point[0] + 0.6)
lat.append(third_point[1] + 0.3)
vel.append(vel_B4)
lon.append(third_point[0] + 0.8)
lat.append(third_point[1] + 0.4)
vel.append(vel_B5)
lon.append(third_point[0] + 1.0)
lat.append(third_point[1] + 0.5)
vel.append(vel_B6)
lon.append(third_point[0] + 1.2)
lat.append(third_point[1] + 0.6)
vel.append(vel_B7)
lon.append(third_point[0] + 1.4)
lat.append(third_point[1] + 0.7)
vel.append(vel_B8)
lon.append(third_point[0] + 1.6)
lat.append(third_point[1] + 0.8)
vel.append(vel_B9)
lon.append(third_point[0] + 1.8)
lat.append(third_point[1] + 0.9)
vel.append(vel_B10)
lon.append(third_point[0] + 2.0)
lat.append(third_point[1] + 1.0)
vel.append(vel_B11)
# Fourth line of points parallel to the third one and further southeast
# runs in the hanging wall side uplift rate values here are parameters
# that will be readjusted during the
# optimize.minimize step
lon.append(fourth_point[0])
lat.append(fourth_point[1])
vel.append(vel_C1)
lon.append(fourth_point[0] + 0.2)
lat.append(fourth_point[1] + 0.1)
vel.append(vel_C2)
lon.append(fourth_point[0] + 0.4)
lat.append(fourth_point[1] + 0.2)
vel.append(vel_C3)
lon.append(fourth_point[0] + 0.6)
lat.append(fourth_point[1] + 0.3)
vel.append(vel_C4)
lon.append(fourth_point[0] + 0.8)
lat.append(fourth_point[1] + 0.4)
vel.append(vel_C5)
lon.append(fourth_point[0] + 1.0)
lat.append(fourth_point[1] + 0.5)
vel.append(vel_C6)
lon.append(fourth_point[0] + 1.2)
lat.append(fourth_point[1] + 0.6)
vel.append(vel_C7)
lon.append(fourth_point[0] + 1.4)
lat.append(fourth_point[1] + 0.7)
vel.append(vel_C8)
lon.append(fourth_point[0] + 1.6)
lat.append(fourth_point[1] + 0.8)
vel.append(vel_C9)
lon.append(fourth_point[0] + 1.8)
lat.append(fourth_point[1] + 0.9)
vel.append(vel_C10)
lon.append(fourth_point[0] + 2.0)
lat.append(fourth_point[1] + 1.0)
vel.append(vel_C11)
# Fifth line of points runs along the AF on the hanging wall
# this line has fixed uplift rates that are close to zero
# fivth_point = [169.53, -44.37]
lon5 = 169.53
lat5 = -44.37
for i in range(10):
    lon5 = lon5 + 0.2
    lon.append(lon5)
    lat5 = lat5 + 0.1
    lat.append(lat5)
    vel.append(0.1)
lon.append(fivth_point[0])
lat.append(fivth_point[1])
vel.append(0.1)

plt.scatter(lon,lat)
plt.show()


z0 = 0.
year = 365.25*24*60*60
z1 = p[-3]
z = np.linspace(z0, z1, 100)

a = []
for i, j in enumerate(lon):
    pred_exh = interpolated_surf(lon[i], lat[i])
    a.append(pred_exh[0])

    temp_prof_par = temperature_profile(z, pred_exh[0]/(1e+3 * year), p)
    z_100 = predict_depth(temp_prof_par, z, 100)/1000*(-1)
    z_200 = predict_depth(temp_prof_par, z, 200)/1000*(-1)
    z_300 = predict_depth(temp_prof_par, z, 300)/1000*(-1)
    z_400 = predict_depth(temp_prof_par, z, 400)/1000*(-1)
    z_500 = predict_depth(temp_prof_par, z, 500)/1000*(-1)
    z_600 = predict_depth(temp_prof_par, z, 600)/1000*(-1)
    print(z_500, pred_exh, (500 - 13) / z_500, (400 - 13) / z_400)

    with open('exhumation_and_temp_grid_points.txt', 'a') as of:
            of.write('{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.
                     format(round(lon[i], 3), round(lat[i], 3),
                            round(z_100, 2), round(z_200, 2),
                            round(z_300, 2), round(z_400, 2),
                            round(z_500, 2), round(z_600, 2),
                            round(pred_exh[0], 2)
                            ))


plt.scatter(lon,lat, c=a, s=500)
plt.show()



    with open('Exhum_grid.txt', 'a') as of:
        of.write('{}, {}, {}\n'.format(round(x[i], 3), round(y[i], 3), pred_exh_per[0]))

# TODO: calculate exhum, T100 etc like above


