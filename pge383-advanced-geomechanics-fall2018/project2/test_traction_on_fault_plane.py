import traction_on_fault_plane as tfp
import geographical_to_principal as gtp
import principal_to_geographical as ptg
import numpy as np
import math

def calc_rotation_matrix(alpha, beta, gamma):
    return gtp.calc_rotation_matrix(math.radians(alpha), \
            math.radians(beta), math.radians(gamma))

def test_2d_normal_faulting():
    rot_pg = calc_rotation_matrix(90, 90, 0)
    sigma_p = np.diag([23, 15, 13.8])
    sigma_g = ptg.rotate_stress_tensor(sigma_p, rot_pg)

    strike = math.radians(0)
    dip = math.radians(60)
    sn, tau = tfp.calc_normal_and_shear_stress(sigma_g, strike, dip)

    assert math.fabs((sn - 16.1)/sn) < 1e-2
    assert math.fabs((tau - 3.98)/tau) < 1e-2
