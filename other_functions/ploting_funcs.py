"""
Read optimization output and make plots.

"""

# Read logfile
path = ('/home/michaiko/Dropbox/Dropbox/VUW/PhD/Codes/'
           'Modeling_1/Latest_codes/modelling_codes/')
log_file = (path + 'logfile_SWISS.txt')
# Read parameters from .csv file
params = read_csv_file(log_file, 36)
# Read bin corners
p1_ = [float(i) for i in params[0]]
p2_ = [float(i) for i in params[1]]
p3_ = [float(i) for i in params[2]]
p4_ = [float(i) for i in params[3]]
p5_ = [float(i) for i in params[4]]
p6_ = [float(i) for i in params[5]]
p7_ = [float(i) for i in params[6]]
p8_ = [float(i) for i in params[7]]
p9_ = [float(i) for i in params[8]]
p10_ = [float(i) for i in params[9]]
p11_ = [float(i) for i in params[10]]
p12_ = [float(i) for i in params[11]]
p13_ = [float(i) for i in params[12]]
p14_ = [float(i) for i in params[13]]
p15_ = [float(i) for i in params[14]]
p16_ = [float(i) for i in params[15]]
p17_ = [float(i) for i in params[16]]
p18_ = [float(i) for i in params[17]]
p19_ = [float(i) for i in params[18]]
p20_ = [float(i) for i in params[19]]
p21_ = [float(i) for i in params[20]]
p22_ = [float(i) for i in params[21]]
p23_ = [float(i) for i in params[22]]
p24_ = [float(i) for i in params[23]]
p25_ = [float(i) for i in params[24]]
p26_ = [float(i) for i in params[25]]
p27_ = [float(i) for i in params[26]]
p28_ = [float(i) for i in params[27]]
p29_ = [float(i) for i in params[28]]
p30_ = [float(i) for i in params[29]]
p31_ = [float(i) for i in params[30]]
p32_ = [float(i) for i in params[31]]
# T1_ = [float(i) for i in params[-1]]
T_BDT_= [float(i) for i in params[-3]]
# Z1_ = [float(i) for i in params[-2]]
it = [float(i) for i in params[-2]]
q = [float(i) for i in params[-1]]

q_ = []
for i, j in enumerate(q):
        aaa = np.log10(j)
        q_.append(aaa)

# Z1 = []
# for i, j in enumerate(Z1_):
#     dep_in_km = j * (-1) /1000
#     Z1.append(dep_in_km)
################################
# Plotting section of the script
################################
font = {'family': 'normal',
        'weight': 'normal',
        'size': 18}
matplotlib.rc('font', **font)
# Set figure width to 12 and height to 9
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size

ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
ax1.plot(it, p1_, zorder=2, label='A1')
ax1.plot(it, p2_, zorder=2,label='A2')
ax1.plot(it, p3_, zorder=2, label='A3')
ax1.plot(it, p4_, zorder=2,label='A4')
ax1.plot(it, p5_, zorder=2, label='A5')
ax1.plot(it, p6_, zorder=2,label='A6')
ax1.plot(it, p7_, zorder=2, label='A7')
ax1.plot(it, p8_, zorder=2,label='A8')
ax1.plot(it, p9_, zorder=2, label='A9')
ax1.plot(it, p10_, zorder=2,label='A10')
ax1.plot(it, p11_, zorder=2, label='A11')
ax1.set_ylabel('Uplift rates (mm/yr)', fontsize=18)
ax1.set_ylim([0, 12])
ax1.set_xlim([0, max(it)])
plt.legend(fontsize=14)

ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
ax2.plot(it, p12_, zorder=2,label='B1')
ax2.plot(it, p13_, zorder=2, label='B2')
ax2.plot(it, p14_, zorder=2,label='B3')
ax2.plot(it, p15_, zorder=2, label='B4')
# ax2.set_ylabel('Uplift rates (mm/yr)', fontsize=18)
ax2.set_ylim([0, 12])
ax2.set_xlim([0, max(it)])
plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)



ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
ax3.plot(it, T_BDT_, zorder=2, label='T_BDT')
# ax2.plot(it, T1_, zorder=2, label='T1')
ax3.set_ylabel(r'Temp($^\circ$C)', fontsize=18)
ax3.set_ylim([0, 700])
ax3.set_xlim([0, max(it)])
plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)
# ax2_ = plt.subplot2grid((4, 1), (2, 0), colspan=1)
# ax2_.plot(it, Z1, zorder=2, label='Z1')
# ax2_.set_ylabel(r'Hyp. depths (km)', fontsize=18)
# # ax2_.set_ylim([0, 700])
# ax2_.set_xlim([0, max(it)])
# plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)
ax4 = plt.subplot2grid((3, 2), (1, 1), colspan=1)
ax4.plot(it, q_, zorder=2, label='Q')
plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)
ax4.set_xlabel(r'Number of iterations', fontsize=20)
ax4.set_xlim([0, max(it)])
ax4.set_ylabel(r'Q', fontsize=20)

plt.show()




# DEBUG
path = ('/home/michaiko/Dropbox/Dropbox/VUW/PhD/Codes/'
           'Modeling_1/Latest_codes/modelling_codes/')
log_file = (path + 'depths.csv')
# Read parameters from .csv file
params = read_csv_file(log_file, 5)
# Read bin corners
itera = [float(i) for i in params[0]]
mod_dep_ = [float(i) for i in params[1]]
obs_dep_ = [float(i) for i in params[2]]
dep_dif_ = [float(i) for i in params[3]]
dep_q = [float(i) for i in params[4]]

dep_dif = []
for i, j in enumerate(dep_dif_):
        if j < 0:
                dep_dif.append(j*(-1)/1000)
        else:
                dep_dif.append(j/1000)
mod_dep = []
for i, j in enumerate(mod_dep_):
    dep_in_km = j * (-1) /1000
    mod_dep.append(dep_in_km)
obs_dep = []
for i, j in enumerate(obs_dep_):
    dep_in_km = j * (-1) /1000
    obs_dep.append(dep_in_km)

ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
ax1.scatter(itera, dep_dif, color='blue', edgecolor='black', alpha=0.4,
            label='Depth difference')
ax1.set_xlim([0, max(itera)])
plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)
ax1.set_ylabel(r'Absolute difference in hyp. depths (m)', fontsize=18)

ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
ax2.scatter(itera, mod_dep, color='yellow',
            alpha=1, label='modelled',zorder=3)
ax2.scatter(itera, obs_dep, color='red',
            alpha=1, label='observed',zorder=2)
ax2.set_xlim([0, max(itera)])
plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)
ax2.set_ylim([0, 30])
plt.gca().invert_yaxis()
ax2.set_ylabel(r'Hyp. depths (km)', fontsize=18)
ax2.set_xlabel(r'Number of iterations', fontsize=20)

ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
ax3.plot(itera, dep_q, zorder=2, label='Q')
plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)
ax3.set_xlabel(r'Number of iterations', fontsize=20)
ax3.set_xlim([0, max(itera)])
ax3.set_ylabel(r'Q', fontsize=20)

plt.show()



################
path = ('/home/michaiko/Dropbox/Dropbox/VUW/PhD/Codes/'
           'Modeling_1/Latest_codes/modelling_codes/')
log_file = (path + 'ages.csv')
# Read parameters from .csv file
params = read_csv_file(log_file, 5)
# Read bin corners
itera = [float(i) for i in params[0]]
mod_age = [float(i) for i in params[1]]
obs_age = [float(i) for i in params[2]]
age_dif = [float(i) for i in params[3]]
age_q = [float(i) for i in params[4]]

ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
ax1.scatter(itera, age_dif, color='blue', edgecolor='black', alpha=0.4,
            label='Ages difference')
ax1.set_xlim([0, max(itera)])
plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)
ax1.set_ylabel(r'Absolute difference in ages (Myr)', fontsize=18)

ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
ax2.scatter(itera, mod_age, color='yellow',
            alpha=1, label='modelled',zorder=3)
ax2.scatter(itera, obs_age, color='red',
            alpha=1, label='observed',zorder=2)
ax2.set_xlim([0, max(itera)])
plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)
ax2.set_ylim([0, 240])
plt.gca().invert_yaxis()
ax2.set_ylabel(r'Ages (Myr)', fontsize=18)
ax2.set_xlabel(r'Number of iterations', fontsize=20)

ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
ax3.plot(itera, age_q, zorder=2, label='Q')
plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)
ax3.set_xlabel(r'Number of iterations', fontsize=20)
ax3.set_xlim([0, max(itera)])
ax3.set_ylabel(r'Q', fontsize=20)
plt.show()

###############################################################################

thermochron_path =('/home/michaiko/Dropbox/Dropbox/VUW/PhD/Codes/'
            'Modeling_1/Latest_codes/modelling_codes/tc_data/')
Tc_path = (thermochron_path)
FT_file = (Tc_path + 'Thermochron_data.csv')
# Read parameters from .csv file
params = read_csv_file(FT_file, 5)
# Read bin corners
X_FT = [float(i) for i in params[0]]
Y_FT = [float(i) for i in params[1]]
Age_FT = [float(i) for i in params[2]]
Err_FT_ = [float(i) for i in params[3]]
Tc_FT = [float(i) for i in params[4]]
Err_FT = []
for i, j in enumerate(Err_FT_):
        error = j
        if error == 0.0:
                error = 0.1
        Err_FT.append(error)


plats = [-43.58, -43.11, -43.41, -43.95, -43.58]
plons = [169.89, 170.85, 171.15, 170.19, 169.89]
p_ = Polygon((np.asarray(zip(plats, plons))))
inside_box = np.array([[X_FT[i], Y_FT[i], Age_FT[i], Err_FT[i], Tc_FT[i]]
                        for i in range(len(Y_FT))
                        if p_.contains(Point(Y_FT[i], X_FT[i]))])
# List of the ZFT data details
x_FT = []
y_FT = []
age_FT = []
err_FT = []
tc_FT = []
for i, j in enumerate(inside_box):
        x_FT.append(j[0])
        y_FT.append(j[1])
        age_FT.append(j[2])
        error = j[3]
        if error == 0.0:
                error = 0.1
                err_FT.append(error)
                tc_FT.append(j[4])
grids_x= []
grids_y= []
for i, j in enumerate(final_uplifts[0]):
        grids_x.append(final_uplifts[1][i])
        grids_y.append(final_uplifts[0][i])

npzfilespath = ('/home/michaiko/Dropbox/Dropbox/VUW/PhD/'
                'Codes/Modeling_1/Latest_codes/modelling_codes/npz_files/')

# npzfilespath = (seismic_path)
# Read npz files and create a list of dictionaries
Final_box_details = read_npz_files(npzfilespath, '*')
x_seis = []
y_seis = []
z_BDT_obs = []
h_unc = []
for i, j in enumerate(Final_box_details):
        x_seis.append(np.mean(j['longi']))
        y_seis.append(np.mean(j['latit']))
        z_BDT_obs.append(j['lower'] * (-1000))
        h_unc.append(np.mean(j['h_unc']) * (1000))


plats_alt = [-43.85, -43.85 + 1, -44.5 + 1, -44.5, -43.85]
plons_alt = [169.1, 169.1+2, 169.7+2, 169.7, 169.1]

ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
ax1.plot(plons, plats, color='blue',  alpha=0.4,
         label='Box')
ax1.plot(plons_alt, plats_alt, color='cyan',  alpha=0.4,
         label='Box_alt')
ax1.scatter(X_FT, Y_FT, color='purple', edgecolor='black', alpha=0.9,
            label='TF data')
ax1.scatter(x_seis, y_seis, color='blue', edgecolor='black', alpha=1,
            label='Seismicity obs')
ax1.scatter(x_FT, y_FT, color='yellow', edgecolor='black', alpha=1,
            label='TF data_used')
ax1.scatter(grids_x, grids_y, color='red', marker='s', edgecolor='black',
            alpha=1, label='Grid points')
plt.legend(loc="lower right", markerscale=1., scatterpoints=1, fontsize=14)
plt.show()


first_point = [169.1, -43.85]
second_point = [169.3, -44.10]
third_point = [169.5, -44.35]
fourth_point = [169.7, -44.6]

plats = [-44, -42.85, -43.225, -44.375]
plons = [168.95, 171.25, 171.55, 169.25]
lon4 = 169.5 + 0.4
lat4 = -44.35 + 0.2
for i in xrange(5):
        lon4 = lon4 + 0.2
        lon.append(lon4)
        lat4 = lat4 + 0.1
        lat.append(lat4)
        vel.append(0.1)
lon.append(fivth_point[0])
lat.append(fivth_point[1])
vel.append(0.1)


seis_lon =[]
seis_lat = []
for i, j in enumerate(seis_obs):
        print j[0]
        seis_lon.append(j[0])
        seis_lat.append(j[1])
ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
ax1.scatter(first_point[0], first_point[1], color='red', marker='s', edgecolor='black', alpha=1,
            label='Grid points')
ax1.scatter(second_point[0], second_point[1], color='red', marker='s', edgecolor='black', alpha=1,
            label='Grid points')
ax1.scatter(third_point[0], third_point[1], color='red', marker='s', edgecolor='black', alpha=1,
            label='Grid points')
ax1.scatter(fourth_point[0], fourth_point[1], color='red', marker='s', edgecolor='black', alpha=1,
            label='Grid points')
# ax1.scatter(fivth_point[0], fivth_point[1], color='green', marker='s', edgecolor='black', alpha=1,
#             label='Grid points')
ax1.scatter(seis_lon, seis_lat, color='orange', marker='s', edgecolor='black', alpha=1,
            label='Grid points')
ax1.scatter(plons, plats, color='blue', marker='s', edgecolor='black', alpha=1,
            label='Grid points')
plt.show()


vel = []
mod_ages = []
w = []
for i, j in enumerate(predicted_exh_tc):
        vel.append(j[0])
        mod_ages.append(modelled_age[i][0])
        w.append(weight[i][0])


ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
ax1.scatter(mod_ages,w)
plt.show()



ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
ax1.scatter(v,w)
plt.show()
