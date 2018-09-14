import geographical_to_principal as gtp
import numpy as np
import math

def calc_rotation_matrix(alpha, beta, gamma):
    return gtp.calc_rotation_matrix(math.radians(alpha), \
            math.radians(beta), math.radians(gamma))

def test_normal_faulting_stress_regime():
    rot = calc_rotation_matrix(0, 90, 0)
    rot_c = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    assert np.linalg.norm(rot - rot_c) < 1e-10

    sigma_p = np.diag([30, 25, 20])
    sigma_g = gtp.rotate_stress_tensor(sigma_p, rot)
    sigma_g_c = np.diag([20, 25, 30])
    assert np.linalg.norm(sigma_g - sigma_g_c) < 1e-10

def test_strike_slip_stress_regime():
    rot = calc_rotation_matrix(0, 0, 90)
    rot_c = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert np.linalg.norm(rot - rot_c) < 1e-10

    sigma_p = np.diag([30, 25, 20])
    sigma_g = gtp.rotate_stress_tensor(sigma_p, rot)
    sigma_g_c = np.diag([30, 20, 25])
    assert np.linalg.norm(sigma_g - sigma_g_c) < 1e-10

def test_reverse_faulting_stress_regime():
    rot = calc_rotation_matrix(90, 0, 0)
    rot_c = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.linalg.norm(rot - rot_c) < 1e-10

    sigma_p = np.diag([30, 25, 20])
    sigma_g = gtp.rotate_stress_tensor(sigma_p, rot)
    sigma_g_c = np.diag([25, 30, 20])
    assert np.linalg.norm(sigma_g - sigma_g_c) < 1e-10
