# -*- coding: utf-8 -*-
"""
PHYS20161 Doppler spectroscopy assignment.

This code reads in and performs a minimised chi squared fit to
analyse the star planet system.

Using the observed wavelength from the star and fitting the velocity of the
star and the angular frequency of the orbit, the orbit radius, planet velocity
and the mass of the planet can be found.

Created on Wed Nov 25 15:31:49 2020

@author: Emily Gillott

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import scipy.constants as pc


# Names of data files
FILE_1 = 'doppler_data_1.csv'
FILE_2 = 'doppler_data_2.csv'

# Define constants
EMITTED_WAVELENGTH = 656.281e-9 # m
INITIAL_PHASE = 0
INITIAL_PERAMATER_GUESS = [50, 3e-8] # velocity, angular frequency
STAR_MASS_SOLAR_MASS = 2.78 # solar masses
SOLAR_MASS = 1.98847e30 # kg
ASTRONOMICAL_UNIT = 1.4960e11 # m
JOVIAN_MASS = 1.89813e27 # kg
ANGLE_OF_ORBIT = np.pi / 2

def read_file(file_name):
    """
    Reads the file input and returns an array of the data.

    Inputs
    -------
    file_name : string

    Returns
    --------
    data_number : numpy array of floats

    Emily Gillott 25/11/2020
    """
    data = np.zeros((0, 3))

    try:
        data = np.genfromtxt(file_name, comments='%', delimiter=',')

        data_number = np.zeros((0, 3))

        for index in range(len(data)):
            if (is_number(data[index, 0]) and is_number(data[index, 1]) and
                    is_number(data[index, 2])):
                data_number = np.vstack((data_number, data[index, :]))
        return data_number

    except OSError:
        print('There was an error locating the file. Please check and try again.')
        sys.exit(1)
        return 1

def is_number(value):
    """
    Checks that an input is a numerical value. Rejects nan and inf values

    Inputs
    --------
    value : the value to be checked

    Returns
    --------
    Bool

    Emily Gillott 25/11/2020
    """
    try:
        float(value)
        return bool(not np.isnan(value))

    except ValueError:
        return False

def order_data(data_1, data_2):
    """
    Returns an array of all the data in ascending order of time.

    Inputs
    --------
    data_1 : numpy array of floats
    data_2 : numpy array of floats

    Returns
    ---------
    data : numpy array of floats

    Emily Gillott 27/11/2020
    """

    data = np.vstack((data_1, data_2))

    indicies = np.argsort(data[:, 0])

    return data[indicies, :]

def validate_data(data_1, data_2):
    """
    Removes any data points which are extreme values (anomolous) and values
    with zero error.

    Inputs
    --------
    data_1 : numpy array
    data_2 : numpy array

    Returns
    --------
    data_clean : numpy array

    Emily Gillott 27/11/2020
    """
    data_all = order_data(data_1, data_2)
    data_all = data_all[np.argwhere(data_all[:, 2] != 0)[:, 0], :]

    # Remove points more than 5 standard deviations away from the mean
    indicies_2 = np.argwhere(np.abs(data_all[:, 1]
                                    - np.mean(data_all[:, 1]))
                             > (5 * np.std(data_all[:, 1])))
    data_clean = np.delete(data_all, indicies_2, axis=0)

    return data_clean

def change_units(data):
    """
    Changes the units of the data array to SI units.

    Inputs
    --------
    data : numpy array (years, nano metres, nano metres)

    Returns
    --------
    data : array of floats (seconds, metres, metres)

    Emily Gillott 04/12/2020
    """
    data = np.array([((data[:, 0] * 365.25 * 24 * 60 * 60),
                      data[:, 1] * 1e-9, data[:, 2] * 1e-9)])

    return data


def get_star_velocity(peak_velocity, angular_frequency, data):
    """
    Finds the velocity of the star given a peak velocity, angular frequency
    and a time.

    Inputs
    --------
    peak_velocity : float
    angular_frequency : float
    data : numpy array

    Returns
    -------
    star_velocity : array if floats

    Emily Gillott 25/11/2020
    """
    star_velocity = (peak_velocity * np.sin(angular_frequency * data[:, 0]
                                            + INITIAL_PHASE)
                     * np.sin(ANGLE_OF_ORBIT))
    return star_velocity

def get_wavelength(peak_velocity, angular_frequency, data):
    """
    Calculates the wavelength given a peak velocity and an angular frequency.

    Inputs
    --------
    peak_velocity : float
    angular_frequency : float
    data : array of floats

    Returns
    --------
    observed_wavelength : array of floats

    Emily Gillott 25/11/2020
    """

    observed_wavelength = ((EMITTED_WAVELENGTH / pc.c)
                           * (pc.c - get_star_velocity(
                               peak_velocity, angular_frequency, data)))

    return observed_wavelength

def chi_squared_1(values):
    """
    Finds the chi square for a given peak velocity and angular frequency

    Inputs
    -------
    values : list containing the peak velocuty and the agular frequency

    Returns
    -------
    chi_squared : float

    Emily Gillott 25/11/2020
    """
    peak_velocity = values[0]
    angular_frequency = values[1]

    prediction = get_wavelength(peak_velocity, angular_frequency, DATA_ALL)

    chi_squared = np.sum(((prediction - DATA_ALL[:, 1]) / DATA_ALL[:, 2]) ** 2)

    return chi_squared

def chi_squared_2(perameters):
    """
    Find the chi square of the clean data file.

    Inputs
    -------
    perameters : list of the peak velocity and angular frequency

    Returns
    --------
    chi_squared : float

    Emily Gillott 08/12/2020
    """
    prediction = get_wavelength(perameters[0], perameters[1], DATA_CLEAN)


    chi_squared = np.sum(((prediction - DATA_CLEAN[:, 1]) / DATA_CLEAN[:, 2])
                         ** 2)

    return chi_squared

def min_chi_square(chi_function):
    """
    Finds the values which minimise the chi squared value.

    Inputs
    -------
    chi_function : the chi square function to be minimised

    Returns

    values : a list of the minimised parameters (peak_velocity and
                                                  angular_frequency)

    Emily Gillott 04/12/2020
    """

    values = fmin(chi_function, INITIAL_PERAMATER_GUESS, disp=False)

    return values

def chi_squared_3():
    """
    Calculates the value of the minimised chi square from the values found in
    the minimisation of the chi square in the final fit.

    Inputs
    --------
    none

    Returns
    --------
    chi_squared : float

    """
    prediction = get_wavelength(PEAK_VELOCITY, ANGULAR_FREQUENCY, DATA_CLEAN)

    chi_squared = np.sum(((prediction - DATA_CLEAN[:, 1]) / DATA_CLEAN[:, 2])
                         **2)

    return chi_squared


def remove_anomolous(peak_velocity, angular_frequency):
    """
    Produces a preliminary fit to the data then looks at the residuals. Removes
    the data points which are more than 5 errors away from the fitted curve.

    Inputs
    -------
    peak_velocity : float
    agular_frequency : float

    Returns
    -------
    data_clean : array of floats

    Emily Gillott 04/12/2020
    """

    prediction = get_wavelength(peak_velocity, angular_frequency, DATA_ALL)

    residuals = DATA_ALL[:, 1] - prediction

    indicies = np.argwhere(np.abs(residuals
                                  - np.mean(residuals))
                           > (5 * DATA_ALL[:, 2]))
    data_clean = np.delete(DATA_ALL, indicies, axis=0)

    return data_clean

def plot_fit():
    """
    Produces a plot of the actual fit (not preliminary fit) to the 'clean'
    data (data where all anomolous values and errors have been removed).

    Inputs
    --------
    none

    Returns
    --------
    none

    Emily Gillott 05/12/2020
    """
    seconds_in_year = 365.25 * 24 * 60 * 60

    prediction = (get_wavelength(PEAK_VELOCITY, ANGULAR_FREQUENCY, DATA_CLEAN)
                  * 10e9)

    residuals = (DATA_CLEAN[:, 1] * 10e9 - prediction)
    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals)

    figure = plt.figure(figsize=(5, 5))
    axis_1 = figure.add_subplot(211)
    axis_1.errorbar(DATA_CLEAN[:, 0] / seconds_in_year, DATA_CLEAN[:, 1] * 10e9,
                    DATA_CLEAN[:, 2] * 10e9, fmt='k.', ls='none', label='Data')
    axis_1.plot(DATA_CLEAN[:, 0] / seconds_in_year, prediction, 'r.',
                label='Fit')
    axis_1.set_title('Data and fit')
    axis_1.set_xlabel('Time (years)')
    axis_1.set_ylabel('Wavelength nm')

    if 0.5 < (MIN_CHI_SQUARED / len(DATA_CLEAN)) < 2:
        figure.text(.1, .5, r' The fit was successful: $\chi_r^2 = {0:.3g}$'
                    .format(MIN_CHI_SQUARED / len(DATA_CLEAN)), fontsize=10)
    else:
        figure.text(.1, .5, r'The fit does not well describe the data' \
                     r'$\chi_r^2 = {0:.3g}$'
                    .format(MIN_CHI_SQUARED / len(DATA_CLEAN)))

    axis_1.legend()

    axis_2 = figure.add_subplot(413)
    axis_2.errorbar(DATA_CLEAN[:, 0] / seconds_in_year, residuals, DATA_CLEAN[:, 2]
                    * 10e9, color='k', ls='none')
    axis_2.plot(DATA_CLEAN[:, 0] / seconds_in_year, (0 * DATA_CLEAN[:, 0]), 'r')
    axis_2.set_title('Residuals')
    axis_2.set_xlabel('Time (years)')

    figure.text(.1, .15, 'Mean: %s' % float('%.4g' % residuals_mean),
                ha='left', size=10)
    figure.text(.1, .1, 'Standard deviation: %s' % float('%.4g'% residuals_std),
                ha='left', size=10)

    plt.tight_layout()
    plt.savefig('fit_of_data.png', dpi=300)
    plt.show()

    return 0

def get_meshes():
    """
    Creates the meshes of the values for the peak velocity of the star and the
    angular frequency of the orbit.

    Inputs
    --------
    none

    Returns
    --------
    values : 2 2D arrays of the values to be plotted on the contour plot.

    Emily Gillott 06/12/2020
    """

    velocity_array = np.linspace((PEAK_VELOCITY * 0.8),
                                 (PEAK_VELOCITY * 1.2), 100)
    frequency_array = np.linspace((ANGULAR_FREQUENCY * 0.9),
                                  (ANGULAR_FREQUENCY * 1.1), 100)

    velocity_array_mesh = np.empty((0, len(velocity_array)))

    for dummy_1 in frequency_array:
        velocity_array_mesh = np.vstack((velocity_array_mesh, velocity_array))

    frequency_array_mesh = np.empty((0, len(frequency_array)))

    for dummy_2 in velocity_array:
        frequency_array_mesh = np.vstack((frequency_array_mesh, frequency_array))

    frequency_array_mesh = np.transpose(frequency_array_mesh)

    values = [velocity_array_mesh, frequency_array_mesh]

    return values

def chi_square_contour():
    """
    Produces a mesh of the chi square values corresponding to the peak velocity
    amdangular frequency of each position in the 2d array.

    Plots a countour plot of the chi square and marks the minimum point on the
    plot.

    Plots the min chi square + 1 sigma contour line and finds errors in the
    peak velocity and the angular frequency.

    Inputs
    -------
    none

    Returns
    --------
    velociry_err : float
    frequency_err :float

    Emily Gillott 08/12/2020
    """
    velocity_mesh, frequency_mesh = get_meshes()

    chi_mesh = np.empty((len(velocity_mesh), len(velocity_mesh[0])))

    for index_1 in range(len(velocity_mesh[0])):
        for index_2 in range(len(frequency_mesh)):

            chi_mesh[index_2, index_1] = chi_squared_2([velocity_mesh[index_2,
                                                                      index_1],
                                                        frequency_mesh[index_2,
                                                                       index_1]])
    levels = [MIN_CHI_SQUARED + 1]

    figure = plt.figure(figsize=(4, 4))

    axis = figure.add_subplot(111)
    contour_line = axis.contour(velocity_mesh, frequency_mesh, chi_mesh, levels,
                                colors='w')


    axis.plot(PEAK_VELOCITY, ANGULAR_FREQUENCY, 'wx',
              label='Minima: {0:.3f}'.format(MIN_CHI_SQUARED))

    filled_contour = axis.contourf(velocity_mesh, frequency_mesh, chi_mesh, 12,
                                   cmap='gist_rainbow')
    figure.colorbar(filled_contour)

    path = contour_line.collections[0].get_paths()[0]
    points = path.vertices

    velocity_error = (np.max(points[:, 0]) - np.min(points[0:, 0])) / 2
    frequency_error = (np.max(points[:, 1]) - np.min(points[0:, 1])) / 2

    axis.set_title('Chi square plot')
    axis.set_xlabel('Peak velocities (m/s)')
    axis.set_ylabel('Angular frequencies (rad/s)')

    plt.tight_layout()
    plt.legend()
    plt.savefig('chi_square_contour_plot.png', dpi=300)
    plt.show()

    return velocity_error, frequency_error

def get_orbit_radius():
    """
    Calculates the separation of the star and the planet.

    Inputs
    -------
    none

    Returns
    --------
    orbit_radius_m : float
    orbit_radius_au : float
    orbit_radius_error_au : float

    Emily Gillott 28/11/2020
    """
    period = 2 * np.pi / ANGULAR_FREQUENCY

    constant = (pc.G * STAR_MASS_KG) / (4 * (np.pi ** 2))

    orbit_radius_m = np.cbrt(constant * (period ** 2))

    orbit_radius_au = orbit_radius_m / pc.au

    period_error = 2 * np.pi * (ANGULAR_FREQUENCY ** (-2)) * ERRORS[1]

    orbit_radius_error_m = ((2 / 3.0) * (constant) ** (1 / 3.0) *
                            (period ** (-1 / 3.0)) * period_error)

    orbit_radius_error_au = orbit_radius_error_m / pc.au

    return orbit_radius_m, orbit_radius_au, orbit_radius_error_au

def get_planet_speed():
    """
    Calculates the speed of the planet.

    Inputs
    -------
    none

    Returns
    --------
    planet_velocity : float
    planet_velocuty_error : float

    Emily Gillott 28/11/2020
    """
    planet_velocity = np.sqrt(pc.G * STAR_MASS_KG
                              / ORBIT_RADIUS[0])

    planet_velocity_error = (np.sqrt((pc.G * STAR_MASS_KG) / (4 * ORBIT_RADIUS[0]))
                             * ORBIT_RADIUS[2])

    return planet_velocity, planet_velocity_error

def get_planet_mass():
    """
    Calculates the mass of the planet

    Inputs
    --------
    none

    Returns
    --------
    planet_mass_kg : float
    planet_mass_jovian : float
    planet_mass_error_jovian : float

    Emily Gillott 28/11/2020
    """
    planet_mass_kg = (np.sqrt(STAR_MASS_KG * get_orbit_radius()[0])
                      * PEAK_VELOCITY) / np.sqrt(pc.G)
    planet_mass_jovian = planet_mass_kg / JOVIAN_MASS

    planet_mass_error_jovian = (np.sqrt(np.square(PLANET_SPEED[1]
                                                  / PLANET_SPEED[0])
                                        + np.square(ERRORS[0] / PEAK_VELOCITY))
                                * planet_mass_jovian)

    return planet_mass_kg, planet_mass_jovian, planet_mass_error_jovian

def get_power_of_ten(value):
    """
    Get's the power of ten of a number.

    Emily Gillott 15/12/2020
    """
    power_of_ten = np.floor(np.log10(value))

    value_root = value / np.power(10, power_of_ten)

    return power_of_ten, value_root

def transpose_formatting(base, target, decimals=3):
    """
    Returns the appropriate mantissa to have formatting between two numbers
    to the same power of 10.

    Parameters
    ----------
    base : float
        Value
    target : float
        Error
    decimals : float

    Returns
    -------
    mantissa : foat

    -------

    From example code : week_7_measure_electron_density
    """
    power_of_ten = np.floor(np.log10(base))

    mantissa = np.round(target / 10 ** power_of_ten, decimals=decimals)

    return mantissa

def print_values():
    """
    Prints value's calculated from the fit to the data

    Inputs
    --------
    none

    Returns
    --------
    none

    Emily Gillott 15/12/2020
    """

    peak_velocity = get_power_of_ten(PEAK_VELOCITY)

    print('Peak velocity: {0:.4g} +/- {1:.3f} e{2} m/s'
          .format(peak_velocity[1], transpose_formatting(PEAK_VELOCITY, ERRORS[0]),
                  peak_velocity[0]))

    angular_frequency = get_power_of_ten(ANGULAR_FREQUENCY)

    print('Angular frequency: {0:.4g} +/- {1:.3f} e{2} rad/s'
          .format(angular_frequency[1],
                  transpose_formatting(ANGULAR_FREQUENCY, ERRORS[1]),
                  angular_frequency[0]))

    orbit_radius = get_power_of_ten(ORBIT_RADIUS[1])

    print('Orbit radius: {0:.4g} +/- {1:.3f} e{2} Au'
          .format(orbit_radius[1],
                  transpose_formatting(ORBIT_RADIUS[1], ORBIT_RADIUS[2]),
                  orbit_radius[0]))

    planet_velocity = get_power_of_ten(PLANET_SPEED[0])

    print('Planet velocity: {0:.4g} +/- {1:.3f} e{2} m/s'
          .format(planet_velocity[1],
                  transpose_formatting(PLANET_SPEED[0], PLANET_SPEED[1]),
                  planet_velocity[0]))

    planet_mass = get_power_of_ten(PLANET_MASS[1])

    print('Planet mass: {0:.4g} +/- {1:.3f} e{2} Jovian masses'
          .format(planet_mass[1],
                  transpose_formatting(PLANET_MASS[1], PLANET_MASS[2]),
                  planet_mass[0]))

    print('Reduced chi squared: {0:.3g} '.format(MIN_CHI_SQUARED
                                                 / (len(DATA_CLEAN) - 2)))

    return 0

STAR_MASS_KG = STAR_MASS_SOLAR_MASS * SOLAR_MASS
DATA_1 = read_file(FILE_1)
DATA_2 = read_file(FILE_2)

DATA_ALL = np.transpose(change_units(validate_data(DATA_1, DATA_2)))[:, :, 0]

FIRST_FIT = min_chi_square(chi_squared_1)

FIRST_PEAK_VELOCITY = FIRST_FIT[0]
FIRST_ANGULAR_FREQUENCY = FIRST_FIT[1]

DATA_CLEAN = remove_anomolous(FIRST_PEAK_VELOCITY, FIRST_ANGULAR_FREQUENCY)

FINAL_FIT = min_chi_square(chi_squared_2)

PEAK_VELOCITY = FINAL_FIT[0]
ANGULAR_FREQUENCY = FINAL_FIT[1]

MIN_CHI_SQUARED = chi_squared_3()

plot_fit()
ERRORS = chi_square_contour()

ORBIT_RADIUS = get_orbit_radius()
PLANET_SPEED = get_planet_speed()
PLANET_MASS = get_planet_mass()

print_values()
