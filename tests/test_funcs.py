import unittest
import numpy as np
from scipy import interpolate

# from Seism_and_thermo import *
from functions import *

vel_A1 = 1.
vel_A2 = 1.
vel_A3 = 1.
vel_A4 = 1.
vel_A5 = 1.
vel_B1 = 1.
vel_B2 = 1.
vel_B3 = 1.
vel_B4 = 1.
vel_B5 = 1. 
z1 = -36000.
p = [vel_A1, vel_A2, vel_A3, vel_A4, vel_A5,
     vel_B1, vel_B2, vel_B3, vel_B4, vel_B5, z1]

# Defines a new test case class called TestSum, 
# which inherits from unittest.TestCase
class Test_interp(unittest.TestCase):
    # Defines a test method, .test_list_int(), 
    # to test a list of integers
    def test_list_interp(self):
        """
        Test that it can ...
        """
        initial_grid = define_grid(p[0], p[1], p[2], p[3], p[4],
                                   p[5], p[6], p[7], p[8], p[9])
        
        f = interp_surf(initial_grid[1], initial_grid[0], 
                                        initial_grid[2])
        vel_B4_coord = [169.5 + 1.2, -44.25 + 0.6]
    
        result = f(vel_B4_coord[0],vel_B4_coord[1])[0]
        # Assert that the value of result equals 6
        # by using the .assertEqual() method on the 
        # unittest.TestCase class 
        self.assertEqual(result, 1)
    
    def test_temperature_profile(self):
        """"""
        year = 365.25*24*60*60
        C = 2700.*790.
        # W/mK Thermal conductivity
        k = 3.5
        # W/m^3s Vol heat productivity - typical value for granite/ heat production
        H = 3.e-6
        z0 = 0.
        T0 = 0.5
        T1 = 550.
        z1 = -36000.
        z = -1200.
        v1 = 8./(1e+3 * year)
        A2 = ((T1-T0) - (z1-z0)*H/(v1*C)) / (np.exp(z1*v1*C/k)-np.exp(z0*v1*C/k))
        A1 = T0 - A2*np.exp(z0*v1*C/k) - z0*H/(v1*C)
        T_test = A1 + A2*np.exp(z*v1*C/k) + z*H/(v1*C)
    
        result = temperature_profile(z, v1, p, C=2700.*790., k=3.5, H=3.e-6, z0=0., T0=0.)
        self.assertEqual(result, T_test)
    
    
    def test_predict_depth(self):
        """"""
        T1 = 550.
        z1 = -36000.
        z0 = 0.
        v1 = 8./(1e+3 * year)
        z = np.linspace(z0, z1, 100)
        t_z = temperature_profile(z, v1, p, C=2700.*790., k=3.5, H=3.e-6, z0=0., T0=0.)
    
        result = predict_depth(t_z, z, T1)
        self.assertEqual(result, z1)


# Defines a command-line entry point, 
# which runs the unittest test-runner .main()
if __name__ == '__main__':
    unittest.main()


# T1 = 550.
# z1 = -35000.
# z0 = 0.

# v1 = 5/(1e+3 * year)
# z = np.linspace(z0, z1, 100)
# t_z = temperature_profile(z, v1, p, C=2700.*790., k=3.5, H=3.e-6, z0=0., T0=0.)
# result = predict_depth(t_z, z, 200)
# print result
# vel_in_mm_yr = v1 * (1e+3 * year)
# z_c_in_km = result * (-1) / 1000
# age = (z_c_in_km/vel_in_mm_yr)  # Myr
# print age

