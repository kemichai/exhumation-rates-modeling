"""Back up functions."""
import csv
import numpy as np
import glob
from shapely.geometry import Polygon, Point
from scipy import interpolate
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


def make_dirs(path):
    """Small function to create directories."""
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory '%s' created" % path.split('/')[-1])
    else:
        print("Directory '%s' exists" % path.split('/')[-1])
    return


# Class used to rotate coordinates of
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


def read_npz_files(path, files):
    """Function to read .npz files."""
    npzfilespath = path
    # Make list of the .npz files
    npz_list = glob.glob(npzfilespath + files)
    # Sort list of .npz files
    npz_list.sort()

    Final_box_details = []
    for i, bin_file in enumerate(npz_list):
        p = np.load(bin_file)
        bin_ind = int(p['index'])
        bin_box_corn = p['box_corners']
        bin_depths = p['hdep']
        bin_lats = p['latit']
        bin_lons = p['longi']
        bin_upper = float(p['Predicted_100'])
        bin_lower = float(p['Predicted_0'])
        bin_r = float(p['Rsquared'])
        bin_slop = float(p['Line_slope'])
        bin_nobs = int(p['Nobs'])
        bin_uncs = p['h_unc']
        bin_lower_unc = float(p['Predicted_0_unc'])
        Final_box_details.append({"index": bin_ind,
                                  "box_corners": bin_box_corn,
                                  "hdep": bin_depths,
                                  "latit": bin_lats,
                                  "longi": bin_lons,
                                  "h_unc": bin_uncs,
                                  "lower": bin_lower,
                                  "lower_unc": bin_lower_unc,
                                  "upper": bin_upper,
                                  "Rsquared": bin_r,
                                  "Line_slope": bin_slop,
                                  "Nobs": bin_nobs})
    return Final_box_details


def define_grid(vel_A1, vel_A2, vel_A3, vel_A4, vel_A5,
                vel_A6, vel_A7, vel_A8, vel_A9, vel_A10, vel_A11,
                vel_B1, vel_B2, vel_B3, vel_B4, vel_B5,
                vel_B6, vel_B7, vel_B8, vel_B9, vel_B10, vel_B11,
                vel_C1, vel_C2, vel_C3, vel_C4, vel_C5,
                vel_C6, vel_C7, vel_C8, vel_C9, vel_C10, vel_C11):
    """
    Defines initial grid for the model setup.

    Creates an initial grid of 4 cross sections along the Alpine Fault (AF)
    with five nodes each. Outer nodes have 0.1 values for
    exhumation. Maximum value is near Aoraki/Mount Cook. We
    do so to test the theory that uplift rates vary significant
    along the lenght of the AF.
    *** BEWARE OF HARD CODING BELOW ***

    :type vel_i: float
    :param vel_i: Uplift rate parameters in 10 nodes along the lenght of the AF

    :returns: A list for longitudes, latitutes, and uplifts
    :rtype: lists
    """
    # Define initial points of the grid in the southwest side of
    # our study area (near Haast)
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

    return lat, lon, vel


def read_observations(thermochron_path, seismic_path):
    """
    Create arrays of the observations.

    Reads both seismicity and thermochronology observations
    with their uncertainties to be used for the modelling calculations.

    :returns: observed seismicity and thermochron data
    :rtype: arrays
    """

    Tc_path = (thermochron_path)

    # This option for the version in the thesis where uncertainties
    # are just the measurement errors
    # FT_file = (Tc_path + 'All_thermochron_data.csv')

    # This option to include the spatial variability of the ages
    # into their uncertainties
    FT_file = (Tc_path + 'Tc_unc.csv')

    params = read_csv_file(FT_file, 6)
    # Longitude (deg)
    lo = [float(i) for i in params[0]]
    # Latitude (deg)
    la = [float(i) for i in params[1]]
    # Thermochronology ages (Myr)
    ag = [float(i) for i in params[2]]
    # Observational errors
    er = [float(i) for i in params[3]]
    # Elevation (m)
    el = [float(i) for i in params[5]]
    # Closure temperature
    tc = [float(i) for i in params[4]]

    thermocron_obs = np.array([[lo[i], la[i], ag[i], er[i],
                                tc[i], el[i]]
                               for i in range(len(lo))])

    #######################
    # Read seismicity data
    #######################
    npzfilespath = (seismic_path)
    # Read .npz files and create a list of dictionaries
    Final_box_details = read_npz_files(npzfilespath, 'params*')
    x_seis = []
    y_seis = []
    z_BDT_obs = []
    h_unc = []
    z_BDT_unc = []
    for i, j in enumerate(Final_box_details):
        x_seis.append(np.mean(j['longi']))
        y_seis.append(np.mean(j['latit']))
        z_BDT_obs.append(j['lower'] * (-1000))
        z_BDT_unc.append(j['lower_unc'] * (1000))
        h_unc.append(np.mean(j['h_unc']) * (1000))
    # Create the array for the seismicity data
    seism_obs = np.array([[x_seis[i], y_seis[i], z_BDT_obs[i], z_BDT_unc[i]]
                         for i in range(len(y_seis))])
    return thermocron_obs, seism_obs


def interp_surf(x_, y_, z_, plot=False, **kwargs):
    """
    Gives the 2-D interpolated surface of exhumation rates.

    Use scipy interpolate to create a 2-D surface of a grid of
    observations.

    :type x_: list
    :param x_: list of coordinates (longitude)
    :type y_: list
    :param y_: list of coordinates (latitutes)
    :type z_: list
    :param z_: list of values to be interpolated
    :param kwargs: Any other arguments accepted by scipy.interpolate

    :returns: interpolated surface
    :rtype: function

    .. rubric:: Example

    >>> f = interp_surf(x,y,z)
    >>> znew = f(xnew, ynew)
    """
    # Convert to arrays
    x = np.asarray(x_)
    y = np.asarray(y_)
    z = np.asarray(z_)

    f = interpolate.interp2d(x, y, z, kind='linear', bounds_error=False)

    if plot:
        # Plot of the surface
        fig = plt.figure(figsize=(10, 6))
        ax = axes3d.Axes3D(fig)
        ax.plot_wireframe(xx, yy, Z)
        ax.plot_surface(xx, yy, Z, cmap=cm.viridis, alpha=0.2)
        ax.scatter3D(x, y, z, c='r')
        ax.set_zlim(.0, 12.0)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        plt.show()
    return f


def temperature_profile(z, v, p, C=2700.*790., k=3.2, H=3.e-6, z0=0., T0=13.):
    """
    Function to calculate the temperatures profiles.

    Computes a steady state T profile for constant exhumation rate and
    fixed boundary conditions.

    :type z: array
    :param z: depth to compute temperature at
    :type v: float
    :param v: exhumation velocity
    :type C: float
    :param C: volumetric heat capacity
    :type k: float
    :param k: thermal conductivity
    :type H: float
    :param H: volumetric heat productivity
    :type z0: float
    :param z0: Top boundary condition (depth)
    :type T0: float
    :param T0: top boundary condition (temp)
    :type z1: float
    :param z1: base boundary condition (depth)
    :type T1: float
    :param T1: base boundary condition (temp)

    :returns: Temperature profile
    :rtype: array
    """
    # Temperature at the bottom of the exhuming block
    T1 = p[-2]
    # Depth of the bottom of the exhuming block
    z1 = p[-3]
    # T1 = 550.
    # z1 = -35000.

    # Solutions of the second order differential equation when
    # imposing the two boundary conditions
    A2 = ((T1-T0) - (z1-z0)*H/(v*C)) / (np.exp(z1*v*C/k)-np.exp(z0*v*C/k))
    A1 = T0 - A2*np.exp(z0*v*C/k) - z0*H/(v*C)

    return A1 + A2*np.exp(z*v*C/k) + z*H/(v*C)


def predict_depth(T_z, z, T_c):
    """
    Estimated predicted depth.

    Uses an estimated temperature profile and a the cooling age
    (according to the type of thermochron data) to calculate a predicted
    depth.

    :type T_z: array
    :param T_z: temperature profile with depth
    :type z: array
    :param z: array of depths (array length same as the one of T_z)
    :type T_c: float
    :param T_c: cooling temperature of the thermochron data used
    :type p: list
    :param p: list of adjustables

    :returns: Predicted depth at T_c temperature
    :rtype: float
    """
    T_cooling = T_c
    # Create two lists (tempdiffs and depths)
    Temp_diffs = []
    for m, temp in enumerate(T_z):
        diff = abs(temp-T_cooling)
        Temp_diffs.append(diff)

    depths = z.tolist()
    # Sort temperature differences and depths at the same time
    sorted_list = [list(x) for x in zip(*sorted(zip(Temp_diffs, depths),
                                        reverse=True,
                                        key=lambda pair: pair[0]))]
    Z_c = sorted_list[-1][-1]
    return Z_c


def predict_age(T_z, z, T_c, pred_vel, datum_elev):
    """
    Predict age using the predicted depth that matches the cooling temperature.

    :type T_z: array
    :param T_z: temperature profile with depth
    :type z: array
    :param z: array of depths (array length same as the one of T_z)
    :type T_c: float
    :param T_c: cooling temperature of the thermochron data used
    :type pred_vel: float
    :param pred_vel: predicted exhumation rate estimated from the interpolated
                    surface
    :type datum_elev: float
    :param datum_elev: elevation of the datum location

    :returns: Predicted age in Myr
    rtype: float
    """
    vel = pred_vel
    year = 365.25*24*60*60

    z_c = predict_depth(T_z, z, T_c)
    vel_in_mm_yr = vel * (1e+3 * year)

    # vel_in_mm_yr = np.asarray([0.0001])
    z_c_in_km = z_c * (-1) / 1000
    elev_in_km = datum_elev / 1000
    # distance from z_c to the elevation where the measurement was made
    comb_distance = z_c_in_km + elev_in_km
    age = (comb_distance/vel_in_mm_yr)  # Myr

    return age


def delta_vel(T_z, z, T_c, datum_elev, obs_age, mod_age):
    """ Calculate the difference in uplifts for the observed and
        modelled ages.
    """
    z_c = predict_depth(T_z, z, T_c)

    z_c_in_km = z_c * (-1) / 1000
    elev_in_km = datum_elev / 1000
    comb_distance = z_c_in_km + elev_in_km

    vel_obs = comb_distance/obs_age
    vel_mod = comb_distance/mod_age
    vel_diff = vel_obs - vel_mod
    # print vel_obs, vel_mod
    return vel_diff


def delta_vel_seis(T_z, z, T_c, datum_elev, obs_age, mod_age):
    """ Calculate the difference in uplifts for the observed and
        modelled ages.
    """
    z_c = predict_depth(T_z, z, T_c)

    z_c_in_km = z_c * (-1) / 1000
    elev_in_km = datum_elev / 1000
    comb_distance = z_c_in_km + elev_in_km

    vel_obs = comb_distance/obs_age
    vel_mod = comb_distance/mod_age
    vel_diff = vel_obs - vel_mod

    return vel_diff


def log_files_thermo(r_thermo_indiv_,
                     pfixed_thermo_, modelled_age_, q_thermo_,
                     vel_difference_, work_dir):
    """ Write files that contain various information
        regarding the residuals of the seismicity and
        thermochron data.
    """
    # write out residuals
    with open(work_dir + '/log_files' +  '/thermo_res.csv', 'a') as of:
        writer = csv.DictWriter(of, fieldnames=["Lon", "Lat", "ther_res",
                                                "vdif", 'elev']
                                , delimiter=',')
        writer.writeheader()
    for i, j in enumerate(r_thermo_indiv_):
        with open(work_dir + '/log_files' + '/thermo_res.csv', 'a') as of:
            of.write('{} {} {} {} {}\n'.
                     format(pfixed_thermo_[i][0], pfixed_thermo_[i][1], j,
                            vel_difference_[i], pfixed_thermo_[i][5]))

    for i, j in enumerate(modelled_age_):
        age_dif = abs(modelled_age_[i][0]) - abs(pfixed_thermo_[i][2])
        with open(work_dir + '/log_files' + '/ages.csv', 'a') as of:
            of.write('{}, {}, {}, {} \n'.
                     format(modelled_age_[i][0], pfixed_thermo_[i][2],
                            age_dif, q_thermo_))
    return


def log_files_seis(r_seis_indiv_, pfixed_seism_,
                   z_BDT_mod_, q_seis_, work_dir):
    """
    Write files that contain various information
    regarding the residuals of the seismicity and
    thermochron data.
    """

    with open(work_dir + '/log_files' + '/seis_res.csv', 'a') as of:
        writer = csv.DictWriter(of, fieldnames=["Lon",
                                                "Lat", "seis_res"]
                                , delimiter=',')
        writer.writeheader()
    for i, j in enumerate(r_seis_indiv_):
        with open(work_dir + '/log_files' + '/seis_res.csv', 'a') as of:
            of.write('{} {} {}\n'.
                     format(pfixed_seism_[i][0], pfixed_seism_[i][1], j))

    for i, j in enumerate(z_BDT_mod_):
        dep_dif = abs(z_BDT_mod_[i]) - abs(pfixed_seism_[i][2])
        with open(work_dir + '/log_files' + '/depths.csv', 'a') as of:
            of.write('{}, {}, {}, {} \n'.
                     format(z_BDT_mod_[i], pfixed_seism_[i][2],
                            dep_dif, q_seis_))
    return


def eq_temp(vel_, z_, p, C=2700.*790., k=3.2, H=3.e-6, z0=0., T0=13.):
    """
    Function to calculate the temperatures of indivual earthquake locations.

    Use the interpolated exhumation rate (vel) and the earthquake's
    hypocentral depth to calculate the temperature of the individual
    earthquake.

    :type vel_: float
    :param vel_: exhumation velocity rate
    :type z_: float
    :param z_: hypocentral depth to compute temperature at
    :type C: float
    :param C: volumetric heat capacity
    :type k: float
    :param k: thermal conductivity
    :type H: float
    :param H: volumetric heat productivity
    :type z0: float
    :param z0: Top boundary condition (depth)
    :type T0: float
    :param T0: top boundary condition (temp)
    :type z1: float
    :param z1: base boundary condition (depth)

    :returns: Temperature at earthquake's hypocentral depth
    :type: float
    """

    # Define year in seconds
    year = 365.25*24*60*60
    # Temperature at the bottom of the exhuming block
    T1 = p[-2]
    # Depth of the bottom of the exhuming block
    z1 = p[-3]
    # Convert the exhumation velocity in mm/yr?
    vel = vel_/year/1000
    # Convert depth from km to meters
    z = z_*(-1000)
    # Solutions of the second order differential equation when
    # imposing the two boundary conditions
    A2 = ((T1-T0) - (z1-z0)*H/(vel*C)) / \
         (np.exp(z1*vel*C/k)-np.exp(z0*vel*C/k))
    A1 = T0 - A2*np.exp(z0*vel*C/k) - z0*H/(vel*C)

    return A1 + A2*np.exp(z*vel*C/k) + z*H/(vel*C)


def plot_temps(uplifts, name, p, res, work_dir, path_for_fig):
    """
    Calculates temperature of each earthquake and plots a histogram.

    Uses the interpolated surface of the exhumation velocities at the
    earthquake location. Then uses this exhumation rate and hypocentral
    depth of the earthquake to calculate the temperature at which the
    earthquake occurrs.

    :type uplifts: array
    :param uplifts: final exhumation rates \
        obtained from the minimization process
    :type name: str
    :param name: Name of model run (used for distinguishing the outputs)
    :type p: list
    :param p: list of adjustable parameters used in the model run
    :type res_: list
    :param res_: summary of the minimization results

    :returns: :class:`matplotlib.figure.Figure`
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    matplotlib.use("Qt5Agg")

    # Use outputs from running the minimization process and
    # define an interpolated surface for the exhumation rates
    interpolated_surf = interp_surf(uplifts[1], uplifts[0],
                                    uplifts[2])
    # Read the individual earthquake locations and in particular
    # the .csv file that contains the vertical uncertainties
    file_name = 'dataset_all.csv'
    eq_cat_file = work_dir + '/dataset/' + file_name
    # Define parameters to be used in the calculations
    params = read_csv_file(eq_cat_file, 12)
    lon = [float(i) for i in params[7]]
    lat = [float(i) for i in params[6]]
    dep = [float(i) for i in params[8]]

    # Create a list of the earthquake hypocentral depths and calculate
    # the temperature in Celcius.
    earthquake_temp_ = []
    earthquake_dep_ = []
    for i, j in enumerate(lon):
        exh = interpolated_surf(lon[i], lat[i])
        eq_t = eq_temp(exh[0], dep[i], p)
        earthquake_temp_.append(eq_t)
        earthquake_dep_.append(dep[i])
    # remove if temps are nan
    earthquake_temp = []
    earthquake_dep = []
    for k, m in enumerate(earthquake_temp_):
        if ~np.isnan(m):
            print(m)
            earthquake_temp.append(m)
            earthquake_dep.append(earthquake_dep_[k])
    # Ploting part of the function
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
    bins = np.arange(-5, 655, 10)
    ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    ax1.hist(earthquake_temp, bins, histtype='step', orientation='vertical',
             color='black', facecolor='grey', alpha=0.9, linewidth=1.5,
             edgecolor='k', fill=True)
    ax1.set_xlim([0, 650])
    ax1.set_ylim([0, 500])
    ax1.set_xlabel(u'Temperature (℃)', fontsize=18)
    ax1.set_ylabel(r'Number of events', fontsize=18)
    plt.axvline(np.median(earthquake_temp), color='k',
                linestyle='dashed', linewidth=2,
                label='Median (' +
                str(round(np.median(earthquake_temp), 1)) + u' ℃)')
    plt.axvline(np.percentile(earthquake_temp, 90),
                color='k', linestyle='dotted', linewidth=2,
                label='90th perc (' +
                str(round(np.percentile(earthquake_temp, 90), 1)) + u' ℃)')
    ax1.axvline(res.x[-1], 0, 90, lw=2, color='r', linestyle='-',
                label=r'T$_{BDT}$ (' + str(round(res.x[-1], 1)) + u' ℃)')

    ax1.legend(loc="upper left", markerscale=1., scatterpoints=1,
               fontsize=17, framealpha=1, borderpad=1)
    ax1.yaxis.set_label_position('right')
    ax1.yaxis.tick_right()
    # Save figure
    # Figure's name
    fig_name = 'eq_temp_' + name + '.png'
    plt.savefig(path_for_fig + '/' + fig_name, bbox_inches="tight", format='png')

    return


def plot_exhum(uplifts, name, path_for_fig):
    """
    Map of contoured exhumation rates.

    Reads the exhumation rate values outputed by the minimization process
    at the grid points of the model and creates a contour of the exhumation
    rates in a map view.

    :type uplifts: array
    :param uplifts: final exhumation rates \
        obtained from the minimization process
    :type name: str
    :param name: Name of model run (used for distinguishing the outputs)

    .. Note::
    Useful link:
    https://stackoverflow.com/questions/26872337/how-can-i-get-my-contour-plot-superimposed-on-a-basemap


    :returns: :class:`matplotlib.figure.Figure`
    """

    import numpy as np
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Qt5Agg")
    from scipy.interpolate import griddata

    # Read the final exhumation values at the grid points
    x_ = []
    y_ = []
    v_ = []
    for i, j in enumerate(uplifts[0]):
        x_.append(uplifts[1][i])
        y_.append(uplifts[0][i])
        v_.append(uplifts[2][i])
    x = np.asarray(x_)
    y = np.asarray(y_)
    v = np.asarray(v_)

    # Create a grid of the data read
    numcols, numrows = 1000, 1000
    xi = np.linspace(x.min(), x.max(), numcols)
    yi = np.linspace(y.min(), y.max(), numrows)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate em
    x, y, z = x, y, v
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # Ploting part of the function
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 14}
    matplotlib.rc('font', **font)
    # Set figure width to 12 and height to 9
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 15
    plt.rcParams["figure.figsize"] = fig_size

    min_lat, max_lat, min_lon, max_lon, min_depth, max_depth = (
            -44.5, -42.5, 168.8, 171.5, 12, 23)

    fig = plt.figure()
    # Map view
    ax1 = fig.add_subplot(1, 1, 1)
    fig.sca(ax1)
    # Define basemap
    bmap = Basemap(llcrnrlon=min_lon, llcrnrlat=min_lat, urcrnrlon=max_lon,
                   urcrnrlat=max_lat, resolution='h', projection='merc',
                   lat_0=min_lat, lon_0=min_lon, ax=ax1, fix_aspect=False)
    bmap.drawcoastlines()
    bmap.drawmapboundary(fill_color='white')
    bmap.fillcontinents(color='white', lake_color='white')
    bmap.drawparallels(np.arange(min_lat, max_lat, 0.5), labels=[1, 0, 0, 0],
                       linewidth=0.5, dashes=[1, 10])
    bmap.drawmeridians(np.arange(min_lon, max_lon, 0.5), labels=[0, 0, 0, 1],
                       linewidth=0.5, dashes=[1, 10])
    xi_, yi_ = bmap(xi, yi)
    conf = bmap.contourf(xi_, yi_, zi, zorder=101, alpha=0.6, cmap='RdPu')
    xx, yy = bmap(x, y)
    # Plot grid points
    bmap.scatter(xx, yy,
                 color='#545454',
                 edgecolor='#ffffff',
                 alpha=.75,
                 s=50,
                 cmap='RdPu',
                 ax=ax1,
                 vmin=z.min(), vmax=z.max(), zorder=101)
    # Define and plot Aoraki/Mount Cook
    AFx, AFy = bmap(170.1410417, -43.5957472)
    bmap.scatter(AFx, AFy,
                 color='k',
                 marker='^',
                 edgecolor='#ffffff',
                 s=50,
                 ax=ax1,
                 zorder=101)
    cbar = plt.colorbar(conf, orientation='horizontal',
                        fraction=.057, pad=0.05)
    cbar.set_label("Exhumation rates - mm/yr")
    bmap.drawmapscale(
        169.3, -42.8, 170, -43,
        50,
        units='km', fontsize=10,
        yoffset=None,
        barstyle='simple', labelstyle='simple',
        fillcolor1='w', fillcolor2='#000000',
        fontcolor='#000000',
        zorder=101)
    # Save figure
    fig_name_exh = 'exh_map_' + name + '.png'
    plt.savefig(path_for_fig + '/' + fig_name_exh, bbox_inches="tight", format='png')
    return


def sort_residuals(file_name, working_directory):
    """
    Function to process residual log file.

    Removes the log outputs from all the iterations apart from the last one.
    """
    import os
    # Write output of residuals
    with open(working_directory + '/' + file_name, 'r') as f:
        for i, line in enumerate(f):

            if line.startswith('Lon'):
                print(line)
                res = []
            else:
                ln = line.split()
                lon = float(ln[0])
                lat = float(ln[1])
                res_sei = float(ln[2])

                a = [lon, lat, res_sei]
                res.append(a)
    # Delete file and write only the results from the last iteration
    filename = working_directory + '/' + file_name
    try:
        os.remove(filename)
    except Exception as e:
        print(e)
        print("Error while deleting file ", filename, ", file doesn't exist!")

    for i, j in enumerate(res):
        print(j)
        with open(working_directory + '/' + file_name, 'a') as of:
            of.write('{} {} {}\n'.
                     format(j[0], j[1], j[2]))

    with open(working_directory + '/' + file_name, 'r') as f:
        lat_ = []
        lon_ = []
        res_ = []
        for i, line in enumerate(f):
            ln = line.split()
            lon_.append(float(ln[0]))
            lat_.append(float(ln[1]))
            res_.append(float(ln[2]))
    residuals = [[] for _ in range(3)]
    residuals[0].append(lon_)
    residuals[1].append(lat_)
    residuals[2].append(res_)
    return residuals


def plot_res(seis_residuals, therm_residuals, name, path_for_fig):
    """
    Map of contoured normalised residuals.

    Reads the exhumation rate values outputed by the minimization process
    at the grid points of the model and creates a contour of the exhumation
    rates in a map view.

    :type seis_residuals: array
    :param seis_residuals:
    :type therm_residuals:
    :param therm_residuals:
    :type name: str
    :param name: Name of model run (used for distinguishing the outputs)

    .. Note::
    Useful link:
    https://stackoverflow.com/questions/26872337/how-can-i-get-my-contour-plot-superimposed-on-a-basemap


    :returns: :class:`matplotlib.figure.Figure`
    """
    import numpy as np
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Qt5Agg")
    from scipy.interpolate import griddata

    # Read seismicity data
    x_seis_ = []
    y_seis_ = []
    v_seis_ = []
    for i, j in enumerate(seis_residuals[0]):
        x_seis_.append(seis_residuals[0][i])
        y_seis_.append(seis_residuals[1][i])
        v_seis_.append(seis_residuals[2][i])

    x_seis = (x_seis_[0])
    y_seis = (y_seis_[0])
    v_seis = (v_seis_[0])

    # Read thermochronology data
    x_therm_ = []
    y_therm_ = []
    v_therm_ = []
    for i, j in enumerate(therm_residuals[0]):
        x_therm_.append(therm_residuals[0][i])
        y_therm_.append(therm_residuals[1][i])
        v_therm_.append(therm_residuals[2][i])

    x_therm = (x_therm_[0])
    y_therm = (y_therm_[0])
    v_therm = (v_therm_[0])

    x = np.asarray(x_seis + x_therm)
    y = np.asarray(y_seis + y_therm)
    v = np.asarray(v_seis + v_therm)

    # Define grid data
    numcols, numrows = 1000, 1000
    xi = np.linspace(x.min(), x.max(), numcols)
    yi = np.linspace(y.min(), y.max(), numrows)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate em
    x, y, z = x, y, v
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # Ploting part of the function
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 14}
    matplotlib.rc('font', **font)
    # Set figure width to 12 and height to 9
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 15
    plt.rcParams["figure.figsize"] = fig_size

    min_lat, max_lat, min_lon, max_lon, min_depth, max_depth = (
            -44.5, -42.5, 168.8, 171.5, 12, 23)

    fig = plt.figure()
    # Map view
    ax1 = fig.add_subplot(1, 1, 1)
    fig.sca(ax1)
    # Define basemap
    bmap = Basemap(llcrnrlon=min_lon, llcrnrlat=min_lat, urcrnrlon=max_lon,
                   urcrnrlat=max_lat, resolution='h', projection='merc',
                   lat_0=min_lat, lon_0=min_lon, ax=ax1, fix_aspect=False)
    bmap.drawcoastlines()
    bmap.drawmapboundary(fill_color='white')
    bmap.fillcontinents(color='white', lake_color='white')
    bmap.drawparallels(np.arange(min_lat, max_lat, 0.5),
                       labels=[1, 0, 0, 0], linewidth=0.5, dashes=[1, 10])
    bmap.drawmeridians(np.arange(min_lon, max_lon, 0.5), labels=[0, 0, 0, 1],
                       linewidth=0.5, dashes=[1, 10])
    xi_, yi_ = bmap(xi, yi)
    conf = bmap.contourf(xi_, yi_, zi, zorder=101, alpha=0.6, cmap='RdBu')
    # Plot seismicity data locations
    xx_seis, yy_seis = bmap(x_seis, y_seis)
    bmap.scatter(xx_seis, yy_seis,
                 color='#545454',
                 edgecolor='#ffffff',
                 alpha=.75,
                 marker='+',
                 label='Seismicity obs.',
                 s=50,
                 cmap='RdBu',
                 ax=ax1,
                 vmin=z.min(), vmax=z.max(), zorder=101)
    # Plot thermochron data locations
    xx_therm, yy_therm = bmap(x_therm, y_therm)
    bmap.scatter(xx_therm, yy_therm,
                 color='#545454',
                 edgecolor='#ffffff',
                 alpha=.75,
                 label='Thermochronology obs',
                 s=50,
                 cmap='RdBu',
                 ax=ax1,
                 vmin=z.min(), vmax=z.max(), zorder=101)
    # Define and plot Aoraki/Mount Cook
    AFx, AFy = bmap(170.1410417, -43.5957472)
    bmap.scatter(AFx, AFy,
                 color='k',
                 marker='^',
                 edgecolor='#ffffff',
                 s=50,
                 ax=ax1,
                 zorder=101)

    cbar = plt.colorbar(conf, orientation='horizontal',
                        fraction=.057, pad=0.05)
    cbar.set_label("Residuals")

    bmap.drawmapscale(
        169.3, -42.8, 170, -43,
        50,
        units='km', fontsize=10,
        yoffset=None,
        barstyle='simple', labelstyle='simple',
        fillcolor1='w', fillcolor2='#000000',
        fontcolor='#000000',
        zorder=101)
    ax1.legend(loc="lower right", markerscale=1., scatterpoints=1,
               fontsize=12, framealpha=1, borderpad=1)
    # Save figure
    fig_name_exh = 'res_map_' + name + '.png'
    plt.savefig(path_for_fig + '/' + fig_name_exh, bbox_inches="tight", format='png')
    return





def temp_cross_section(final_uplifts, p, working_dir):
    """Creates temperature isotherms to be plotted in cross section"""
    import os
    interpolated_surf = interp_surf(final_uplifts[1], final_uplifts[0],
                                    final_uplifts[2])
    work_dir = working_dir
    # This creates the input for the cross section plot
    filenames = [work_dir + '/manuscript_plots/Figure_9/temps_par.txt',
                 work_dir + '/manuscript_plots/Figure_10/temps_per.txt']
    # Loop through the files and delete them if they already exist
    for filepath in filenames:
        try:
            os.remove(filepath)
        except Exception as e:
            print(e)
            print("Error while deleting file ", filepath, ", file doesn't exist!")

    #######################################################
    # Read the locations of a cross section running parallel
    # to the strike of the Alpine Fault and calculate the
    # exhumation rates along this profile as well as the
    # 100, ..., 500 centigrade depths
    plot_dir = (work_dir + '/manuscript_plots')
    file_name_par = '/Figure_9/cross_section.csv'
    cs_file_par = plot_dir + file_name_par
    #
    params = read_csv_file(cs_file_par, 2)
    cs_lon_par = [float(i) for i in params[0]]
    cs_lat_par = [float(i) for i in params[1]]

    z0 = 0.
    year = 365.25*24*60*60
    z1 = p[-3]
    z = np.linspace(z0, z1, 100)

    for i, j in enumerate(cs_lon_par):
        pred_exh_par = interpolated_surf(cs_lon_par[i], cs_lat_par[i])
        temp_prof_par = temperature_profile(z, pred_exh_par[0]/(1e+3 * year), p)
        z_100 = predict_depth(temp_prof_par, z, 100)/1000*(-1)
        z_200 = predict_depth(temp_prof_par, z, 200)/1000*(-1)
        z_300 = predict_depth(temp_prof_par, z, 300)/1000*(-1)
        z_400 = predict_depth(temp_prof_par, z, 400)/1000*(-1)
        z_500 = predict_depth(temp_prof_par, z, 500)/1000*(-1)
        z_600 = predict_depth(temp_prof_par, z, 600)/1000*(-1)
        print(z_500, pred_exh_par, (500 - 13) / z_500, (400 - 13) / z_400)
        # Create a file that can then be used with GMT to plot the
        # isotherms in cross-sections
        with open(plot_dir + '/Figure_9/temps_par.txt', 'a') as of:
            of.write('{} {} {} {} {} {} {} {} {}\n'.
                     format(cs_lon_par[i], cs_lat_par[i],
                            round(z_100, 2), round(z_200, 2),
                            round(z_300, 2), round(z_400, 2),
                            round(z_500, 2), round(z_600, 2),
                            round(pred_exh_par[0], 2)
                            ))

    # same for perpendicular to the AF cross section in Figure 10
    file_name_per = '/Figure_10/cross_section_per.csv'
    cs_file_per = plot_dir + file_name_per
    #
    params = read_csv_file(cs_file_per, 2)
    cs_lon_per = [float(i) for i in params[0]]
    cs_lat_per = [float(i) for i in params[1]]

    for i, j in enumerate(cs_lon_per):
        pred_exh_per = interpolated_surf(cs_lon_per[i], cs_lat_per[i])
        temp_prof_per = temperature_profile(z, pred_exh_per[0]/(1e+3 * year), p)
        z_100 = predict_depth(temp_prof_per, z, 100)/1000*(-1)
        z_200 = predict_depth(temp_prof_per, z, 200)/1000*(-1)
        z_300 = predict_depth(temp_prof_per, z, 300)/1000*(-1)
        z_400 = predict_depth(temp_prof_per, z, 400)/1000*(-1)
        z_500 = predict_depth(temp_prof_per, z, 500)/1000*(-1)
        z_600 = predict_depth(temp_prof_per, z, 600)/1000*(-1)
        print(z_500, pred_exh_per, (500 - 13) / z_500, (400 - 13) / z_400)
        # Create a file that can then be used with GMT to plot the
        # isotherms in cross-sections
        with open(plot_dir + '/Figure_10/temps_per.txt', 'a') as of:
            of.write('{} {} {} {} {} {} {} {} {}\n'.
                     format(cs_lon_per[i], cs_lat_per[i],
                            round(z_100, 2), round(z_200, 2),
                            round(z_300, 2), round(z_400, 2),
                            round(z_500, 2), round(z_600, 2),
                            round(pred_exh_per[0], 2)
                            ))
    return


def Q_steady_state(p, fixed_parameters):
    """
    Function to calculate the total misfit between the observed and predicted
    hypocentral depths (seismicity data) and age (thermochronological data).

    Takes in two lists one with the adjustable and another one with the fixed
    parameters and calculates the misfit to the predicted values on a 1-D
    thermal structure model assuming a steady-state exhumation geotherm.

    :type p: array
    :param p: list of adjustable/varying parameters
    :type fixed_parameters: array
    :param fixed_parameters: list of fixed parameters

    :returns: The total misfit value
    :rtype: float
    """

    # Create initial grid
    initial_grid = define_grid(p[0], p[1], p[2], p[3], p[4],
                               p[5], p[6], p[7], p[8], p[9],
                               p[10], p[11], p[12], p[13],
                               p[14], p[15], p[16], p[17], p[18],
                               p[19], p[20], p[21], p[22], p[23],
                               p[24], p[25], p[26], p[27], p[28],
                               p[29], p[30], p[31], p[32])
    # Calculate interpolated surface
    interpolated_surf = interp_surf(initial_grid[1],
                                    initial_grid[0],
                                    initial_grid[2])
    # Year in seconds
    year = 365.25*24*60*60
    # Temperature at the bottom of the exhuming block (fixed)
    T1 = fixed_parameters[0]
    # Temperature at the bottom of the exhuming block (allowed to vary)
    # T1 = p[-2]
    # Depth of the bottom of exhuming block (fixed)
    z1 = fixed_parameters[1]
    # Depth of the bottom of exhuming block (allowed to vary)
    # z1 = p[-3]
    # Sea level (top of exhuming block)
    z0 = 0.

    # Sort out output paths
    work_dir = fixed_parameters[-2]
    run_name = fixed_parameters[-1]
    output_path = os.path.join(work_dir, 'outputs', run_name)

    # Read seismicity and thermochron observations
    pfixed_thermo = fixed_parameters[2]
    pfixed_seism = fixed_parameters[3]
    alpha = fixed_parameters[-3]

    # Temperature at the britle-ductile temperature
    T_BDT = p[-1]
    # Numpy array of depths between the top and bottom of the exhuming block
    z = np.linspace(z0, z1, 100)

    # Thermochron data
    predicted_exh_tc = [interpolated_surf(pfixed_thermo[i][0],
                                          pfixed_thermo[i][1])
                        for i in range(len(pfixed_thermo))]
    # Steady state-state exhumation geotherm
    T_prof = [temperature_profile(z, predicted_exh_tc[i]/(1e+3 * year), p)
              for i in range(len(predicted_exh_tc))]
    # Calculate the predicted ages
    modelled_age = [predict_age(T_prof[i], z, pfixed_thermo[i][4],
                    predicted_exh_tc[i]/(1e+3 * year), pfixed_thermo[i][5])
                    for i in range(len(T_prof))]
    # Individual thermochron observation misfits (this is how q_thermo was
    # calculated in the submitted thesis
    q_thermo_indiv = [(((pfixed_thermo[i][2] -
                      modelled_age[i][0]))**2)/(pfixed_thermo[i][3]**2)
                      for i in range(len(T_prof))]
    # Residuals of thermochron observations
    r_thermo_indiv = [(((pfixed_thermo[i][2] -
                      modelled_age[i][0])/pfixed_thermo[i][3]))
                      for i in range(len(T_prof))]
    q_thermo = np.sum(q_thermo_indiv)
    # Delta velocity (needs to be checked)
    vel_difference_therm = [delta_vel(T_prof[i], z, pfixed_thermo[i][4],
                            pfixed_thermo[i][5], pfixed_thermo[i][2],
                            modelled_age[i][0]) for i in range(len(T_prof))]

    # Seismicity observations
    predicted_exh_sei = [interpolated_surf(pfixed_seism[i][0],
                                           pfixed_seism[i][1])
                         for i in range(len(pfixed_seism))]
    T_prof_seis = [temperature_profile(z, predicted_exh_sei[i]/(1e+3 * year),
                                       p)
                   for i in range(len(predicted_exh_sei))]
    # Predicted depth of the brittle-ductile zone
    z_BDT_mod = [predict_depth(T_prof_seis[i], z, T_BDT)
                 for i in range(len(T_prof_seis))]
    # Individual seismicity observation misfits (this is how q_seis
    # was calculated in the submitted thesis
    q_seis_indiv = [(((pfixed_seism[i][2] -
                    z_BDT_mod[i]))**2)/(pfixed_seism[i][3]**2)
                    for i in range(len(T_prof_seis))]
    # Residuals
    r_seis_indiv = [(((pfixed_seism[i][2] -
                    z_BDT_mod[i])/(pfixed_seism[i][3])))
                    for i in range(len(T_prof_seis))]
    q_seis = np.sum(q_seis_indiv)

    # Write out thermochron residuals
    log_files_thermo(r_thermo_indiv,
                     pfixed_thermo, modelled_age,
                     q_thermo, vel_difference_therm, output_path)
    # Write out seismicity residuals
    log_files_seis(r_seis_indiv,
                   pfixed_seism, z_BDT_mod,
                   q_seis, output_path)

    print(alpha, alpha*q_thermo, q_seis)
    q = q_seis + q_thermo * alpha
    return q


def Q_initial_state(p, fixed_parameters):
    """
    Function to calculate the total misfit between the observed and predicted
    hypocentral depths (seismicity data) and age (thermochronological data).

    Takes in two lists one with the adjustable and another one with the fixed
    parameters and calculates the misfit to the predicted values on a 1-D
    thermal structure model assuming an initial stable geotherm with non
    exhuming crust.

    :type p: array
    :param p: list of adjustable/varying parameters
    :type fixed_parameters: array
    :param fixed_parameters: list of fixed parameters

    :returns: The total misfit value
    :rtype: float
    """

    # Create initial grid
    initial_grid = define_grid(p[0], p[1], p[2], p[3], p[4],
                               p[5], p[6], p[7], p[8], p[9],
                               p[10], p[11], p[12], p[13],
                               p[14], p[15], p[16], p[17], p[18],
                               p[19], p[20], p[21], p[22], p[23],
                               p[24], p[25], p[26], p[27], p[28],
                               p[29], p[30], p[31], p[32])
    # Calculate interpolated surface
    interpolated_surf = interp_surf(initial_grid[1],
                                    initial_grid[0],
                                    initial_grid[2])
    # Year in seconds
    year = 365.25*24*60*60
    # Temperature at the bottom of the exhuming block (fixed)
    T1 = fixed_parameters[0]
    # Temperature at the bottom of the exhuming block (allowed to vary)
    # T1 = p[-2]
    # Depth of the bottom of exhuming block (fixed)
    z1 = fixed_parameters[1]
    # Depth of the bottom of exhuming block (allowed to vary)
    # z1 = p[-3]
    # Sea level (top of exhuming block)
    z0 = 0.

    # Sort out output paths
    work_dir = fixed_parameters[-2]
    run_name = fixed_parameters[-1]
    output_path = os.path.join(work_dir, 'outputs', run_name)

    # Read seismicity and thermochron observations
    pfixed_thermo = fixed_parameters[2]
    pfixed_seism = fixed_parameters[3]
    alpha = fixed_parameters[-3]

    # Temperature at the britle-ductile temperature
    T_BDT = p[-1]
    # Numpy array of depths between the top and bottom of the exhuming block
    z = np.linspace(z0, z1, 100)

    # Thermochron data
    predicted_exh_tc = [interpolated_surf(pfixed_thermo[i][0],
                                          pfixed_thermo[i][1])
                        for i in range(len(pfixed_thermo))]
    # Initial model (initial stable geotherm with non exhuming crust.)
    T_prof = [temperature_profile(z, np.asarray([0.0001])/(1e+3 * year), p)
              for i in range(len(predicted_exh_tc))]
    # Calculate the predicted ages
    modelled_age = [predict_age(T_prof[i], z, pfixed_thermo[i][4],
                    predicted_exh_tc[i]/(1e+3 * year), pfixed_thermo[i][5])
                    for i in range(len(T_prof))]
    # Individual thermochron observation misfits (this is how q_thermo was
    # calculated in the submitted thesis
    q_thermo_indiv = [(((pfixed_thermo[i][2] -
                      modelled_age[i][0]))**2)/(pfixed_thermo[i][3]**2)
                      for i in range(len(T_prof))]
    # Residuals of thermochron observations
    r_thermo_indiv = [(((pfixed_thermo[i][2] -
                      modelled_age[i][0])/pfixed_thermo[i][3]))
                      for i in range(len(T_prof))]
    q_thermo = np.sum(q_thermo_indiv)
    # Delta velocity (needs to be checked)
    vel_difference_therm = [delta_vel(T_prof[i], z, pfixed_thermo[i][4],
                            pfixed_thermo[i][5], pfixed_thermo[i][2],
                            modelled_age[i][0]) for i in range(len(T_prof))]

    # Seismicity observations
    predicted_exh_sei = [interpolated_surf(pfixed_seism[i][0],
                                           pfixed_seism[i][1])
                         for i in range(len(pfixed_seism))]
    T_prof_seis = [temperature_profile(z, predicted_exh_sei[i]/(1e+3 * year),
                                       p)
                   for i in range(len(predicted_exh_sei))]
    # Predicted depth of the brittle-ductile zone
    z_BDT_mod = [predict_depth(T_prof_seis[i], z, T_BDT)
                 for i in range(len(T_prof_seis))]
    # Individual seismicity observation misfits (this is how q_seis
    # was calculated in the submitted thesis
    q_seis_indiv = [(((pfixed_seism[i][2] -
                    z_BDT_mod[i]))**2)/(pfixed_seism[i][3]**2)
                    for i in range(len(T_prof_seis))]
    # Residuals
    r_seis_indiv = [(((pfixed_seism[i][2] -
                    z_BDT_mod[i])/(pfixed_seism[i][3])))
                    for i in range(len(T_prof_seis))]
    q_seis = np.sum(q_seis_indiv)

    # Write out thermochron residuals
    log_files_thermo(r_thermo_indiv,
                     pfixed_thermo, modelled_age,
                     q_thermo, vel_difference_therm, output_path)
    # Write out seismicity residuals
    log_files_seis(r_seis_indiv,
                   pfixed_seism, z_BDT_mod,
                   q_seis, output_path)

    print(alpha, alpha*q_thermo, q_seis)
    q = q_seis + q_thermo * alpha
    return q


def Q_weights(p, fixed_parameters):
    """
    NEED TO WORK on this function.

    :type p: array
    :param p: list of adjustable/varying parameters
    :type fixed_parameters: array
    :param fixed_parameters: list of fixed parameters

    :returns: The total misfit value
    :rtype: float
    """

    # Create initial grid
    initial_grid = define_grid(p[0], p[1], p[2], p[3], p[4],
                               p[5], p[6], p[7], p[8], p[9],
                               p[10], p[11], p[12], p[13],
                               p[14], p[15], p[16], p[17], p[18],
                               p[19], p[20], p[21], p[22], p[23],
                               p[24], p[25], p[26], p[27], p[28],
                               p[29], p[30], p[31], p[32])
    # Calculate interpolated surface
    interpolated_surf = interp_surf(initial_grid[1],
                                    initial_grid[0],
                                    initial_grid[2])
    # Year in seconds
    year = 365.25*24*60*60
    # Temperature at the bottom of the exhuming block (fixed)
    T1 = fixed_parameters[0]
    # Temperature at the bottom of the exhuming block (allowed to vary)
    # T1 = p[-2]
    # Depth of the bottom of exhuming block (fixed)
    z1 = fixed_parameters[1]
    # Depth of the bottom of exhuming block (allowed to vary)
    # z1 = p[-3]
    # Sea level (top of exhuming block)
    z0 = 0.

    # Sort out output paths
    work_dir = fixed_parameters[-2]
    run_name = fixed_parameters[-1]
    output_path = os.path.join(work_dir, 'outputs', run_name)

    # Read seismicity and thermochron observations
    pfixed_thermo = fixed_parameters[2]
    pfixed_seism = fixed_parameters[3]
    alpha = fixed_parameters[-3]

    # Temperature at the britle-ductile temperature
    T_BDT = p[-1]
    # Numpy array of depths between the top and bottom of the exhuming block
    z = np.linspace(z0, z1, 100)

    # Thermochron data
    predicted_exh_tc = [interpolated_surf(pfixed_thermo[i][0],
                                          pfixed_thermo[i][1])
                        for i in range(len(pfixed_thermo))]
    # Initial model (initial stable geotherm with non exhuming crust.)
    T_prof = [temperature_profile(z, np.asarray([0.0001])/(1e+3 * year), p)
              for i in range(len(predicted_exh_tc))]
    # ------------------------------------------------------
    modelled_age = [predict_age(T_prof[i], z, pfixed_thermo[i][4],
                    predicted_exh_tc[i]/(1e+3 * year), pfixed_thermo[i][5])
                    for i in range(len(T_prof))]
############################################################################
#   We apply a weight to all the observations in order to make the
#   obs have equal ...
#   (that is calculated by ...)
#   because we do not know the uncertainties of these observations.
    # weight = [predicted_exh_tc[i]**2/modelled_age[i]**2
    #           for i in range(len(predicted_exh_tc))]
    # q_thermo_indiv = [((weight[i] * ((pfixed_thermo[i][2] - modelled_age[i][0]))**2))
    #                   for i in range(len(T_prof))]
###############################################################################
#     This is how q_thermo was calculated in the submitted thesis
    q_thermo_indiv = [(((pfixed_thermo[i][2] - modelled_age[i][0]))**2)/(pfixed_thermo[i][3]**2)
                      for i in range(len(T_prof))]
###############################################################################

    r_thermo_indiv = [(((pfixed_thermo[i][2] - modelled_age[i][0])/pfixed_thermo[i][3]))
                      for i in range(len(T_prof))]
    q_thermo = np.sum(q_thermo_indiv)
    # Delta velocity
    vel_difference_therm = [delta_vel(T_prof[i], z, pfixed_thermo[i][4],
                            pfixed_thermo[i][5], pfixed_thermo[i][2],
                            modelled_age[i][0]) for i in range(len(T_prof))]

    # Seismicity observations
    predicted_exh_sei = [interpolated_surf(pfixed_seism[i][0],
                                           pfixed_seism[i][1])
                         for i in range(len(pfixed_seism))]
    T_prof_seis = [temperature_profile(z, predicted_exh_sei[i]/(1e+3 * year),
                                       p)
                   for i in range(len(predicted_exh_sei))]
    z_BDT_mod = [predict_depth(T_prof_seis[i], z, T_BDT)
                 for i in range(len(T_prof_seis))]
###############################################################################
#     This is how q_seis was calculated in the submitted thesis
    q_seis_indiv = [(((pfixed_seism[i][2] - z_BDT_mod[i]))**2)/(pfixed_seism[i][3]**2)
                    for i in range(len(T_prof_seis))]
###############################################################################

    # def calc_w(v):
    #     delta_v = 0.1
    #     v = np.arange(-3.55, 15.55, delta_v)
    #     T_profs = [temperature_profile(z, v[i]/(1e+3 * year), p)
    #                for i in range(len(v))]
    #     z_b = [predict_depth(T_profs[i], z, T_BDT)
    #            for i in range(len(T_profs))]
    #     delta_z = np.asarray([(z_b[i+1] - z_b[i-1])/2
    #                          for i in range(len(z_b)-1)])
    # # delta_z = [np.diff(z_b)
    # #            for i in range(len(T_prof_seis))]
    #
    #     weight = [(delta_v/delta_z[i])**2 for i in range(len(delta_z))]
    #     return weight
    # # w = [(delta_v/delta_z[i])**2 for i in range(len(delta_z))]
    #
    # w = calc_w(predicted_exh_sei)
    #
    # # this is the latest way to calculate q_seismo
    # q_seis_indiv = [(w[i] * (pfixed_seism[i][2] - z_BDT_mod[i])**2)
    #                 for i in range(len(T_prof_seis))]

    ###########################################################################

    r_seis_indiv = [(((pfixed_seism[i][2] - z_BDT_mod[i])/(pfixed_seism[i][3])))
                    for i in range(len(T_prof_seis))]
    q_seis = np.sum(q_seis_indiv)


    # write out residuals
    log_files_thermo(r_thermo_indiv,
                     pfixed_thermo, modelled_age,
                     q_thermo, vel_difference_therm, output_path)
    # # write out residuals
    log_files_seis(r_seis_indiv,
                   pfixed_seism, z_BDT_mod,
                   q_seis, output_path)

    print(alpha, alpha*q_thermo, q_seis)
    q = q_seis + q_thermo * alpha
    return q
