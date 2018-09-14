import numpy as np
import math
import random
import coordinate_transformation as ct
import mohr_circle as mc
from matplotlib import pyplot as plt

def generate_fractures(number_of_fractures):
    strikes = np.random.uniform(0, math.radians(360), number_of_fractures)
    dips = np.random.uniform(0, math.radians(90), number_of_fractures)
    return strikes, dips

def calc_normal_and_shear_components(t, nd, ns, nn):
    sn = np.dot(t, nn)
    td = np.dot(t, nd)
    ts = np.dot(t, ns)
    tau = math.sqrt(td ** 2 + ts ** 2)
    return sn, tau

def calc_normal_and_shear_stress(sigma_g, pore_pressure, strikes, dips):
    assert strikes.size == dips.size

    fracture_plane = ct.FaultPlane(0, 0)
    sn_array = np.zeros(strikes.size)
    tau_array = np.zeros(strikes.size)

    for i in range(len(strikes)):
        fracture_plane.set(strikes[i], dips[i])
        nd, ns, nn = fracture_plane.calc_bases()
        t = np.dot(sigma_g, nn)
        sn, tau = calc_normal_and_shear_components(t, nd, ns, nn)
        sn_array[i] = sn - pore_pressure
        tau_array[i] = tau

    return sn_array, tau_array


if __name__=='__main__':
    """
    Problem 1
    """

    # ------------------------ Data at Depth E --------------------------------
    shmax = 11400
    shmin = 9300
    sv = 9600
    sp = np.array([shmax, shmin, sv])
    pore_pressure = 8000
    # Principal stress tensor, [SHmax, Shmin, Sv]
    sigma_p = np.diag(sp)
    angles = ct.EulerAngles(90, 0, 0, unit="degrees")

    rot_pg = angles.calc_rotation_matrix()
    # Geographical stress tensor
    sigma_g = ct.rotate_tensor(sigma_p, np.transpose(rot_pg))

    print('sigma_p', sigma_p, sep='\n')
    print('sigma_g', sigma_g, sep='\n')

    """
    Problem 2
    """

    # Calculate normal and shear stress acting on randomly distributed fracture
    # planes
    number_of_fractures = 100
    strikes, dips = generate_fractures(number_of_fractures)
    sn, tau = calc_normal_and_shear_stress(sigma_g, pore_pressure, strikes, dips)

    # Calculate Mohr circle
    mohr_circle = mc.MohrCircle(shmax, sv, shmin, pore_pressure)
    circle_x, circle_y = mohr_circle.calc_circle()

    # Calculate Mohr-Coulomb failure envelope
    mu_upper = 0.6
    mu_lower = 0.4
    line_x = np.array([0, shmax - pore_pressure])
    line_y_upper = mu_upper*line_x
    line_y_lower = mu_lower*line_x

    # Plot
    plt.plot(sn, tau, 'o', circle_x, circle_y, '-', line_x, line_y_upper, '-', \
            line_x, line_y_lower, '-')
    plt.xlabel('sigma_n')
    plt.ylabel('tau')
    plt.legend(['fracture', 'Mohr circle', 'failure envelope'])

    plt.show()
