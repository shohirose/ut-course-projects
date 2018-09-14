import coordinate_transformation as ct
import numpy as np
import math

def test_normal_faulting_stress_regime():
    angles = ct.EulerAngles(0, 90, 0, unit="degrees")
    rot_pg = angles.calc_rotation_matrix()
    rot = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    assert np.linalg.norm(rot_pg - rot) < 1e-10

    sigma_p = np.diag([30, 25, 20])
    sigma_g = ct.rotate_tensor(sigma_p, np.transpose(rot_pg))
    sigma_gc = np.diag([20, 25, 30])
    assert np.linalg.norm(sigma_g - sigma_gc) < 1e-10

def test_strike_slip_stress_regime():
    angles = ct.EulerAngles(0, 0, 90, unit="degrees")
    rot_pg = angles.calc_rotation_matrix()
    rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert np.linalg.norm(rot_pg - rot) < 1e-10

    sigma_p = np.diag([30, 25, 20])
    sigma_g = ct.rotate_tensor(sigma_p, np.transpose(rot_pg))
    sigma_gc = np.diag([30, 20, 25])
    assert np.linalg.norm(sigma_g - sigma_gc) < 1e-10

def test_reverse_faulting_stress_regime():
    angles = ct.EulerAngles(90, 0, 0, unit="degrees")
    rot_pg = angles.calc_rotation_matrix()
    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.linalg.norm(rot_pg - rot) < 1e-10

    sigma_p = np.diag([30, 25, 20])
    sigma_g = ct.rotate_tensor(sigma_p, np.transpose(rot_pg))
    sigma_gc = np.diag([25, 30, 20])
    assert np.linalg.norm(sigma_g - sigma_gc) < 1e-10

def test_2d_normal_faulting():
    angles = ct.EulerAngles(90, 90, 0, unit="degrees")
    rot_pg = angles.calc_rotation_matrix()
    sigma_p = np.diag([23, 15, 13.8])
    sigma_g = ct.rotate_tensor(sigma_p, np.transpose(rot_pg))
    sigma_gc = np.diag([15, 13.8, 23])
    assert np.linalg.norm(sigma_g - sigma_gc) < 1e-10

    fault = ct.FaultPlane(0, 60, unit="degrees")
    nd, ns, nn = fault.calc_bases()
    assert np.linalg.norm(nn - np.array([0, 0.867, -0.5])) < 1e-2

    t = np.dot(sigma_g, nn)
    assert np.linalg.norm(t - np.array([0, 11.95, -11.5])) < 1e-2

    sn = np.dot(t, nn)
    td = np.dot(t, nd)
    ts = np.dot(t, ns)
    tau = math.sqrt(td ** 2 + ts ** 2)
    assert math.fabs((sn - 16.1)/sn) < 1e-2
    assert math.fabs((td + 3.98)/td) < 1e-2
    assert math.fabs(ts) < 1e-2

