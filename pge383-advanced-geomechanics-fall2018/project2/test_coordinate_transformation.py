import coordinate_transformation as ct
import numpy as np
import math

def calc_normal_and_shear_components(traction, nd, ns, nn):
    """
    Calculate normal and shear components of traction.

    Parameters
    ----------
    traction : numpy.array
        Traction vector
    nd : numpy.array
        Unit vector in the dip direction
    ns : numpy.array
        Unit vector in the strike direction
    nn : numpy.array
        Unit vector in the fault normal direction

    Returns
    -------
    sn : double
        Normal component of the traction on a fault plane
    tau : double
        Shear component of the traction on a fault plane
    """

    sn = np.dot(traction, nn)
    td = np.dot(traction, nd)
    ts = np.dot(traction, ns)
    tau = math.sqrt(td ** 2 + ts ** 2)
    return sn, tau

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
    assert np.linalg.norm(sigma_g - sigma_gc)/np.amax(np.fabs(sigma_g)) < 1e-3

    fault = ct.FaultPlane(0, 60, unit="degrees")
    nd, ns, nn = fault.calc_bases()
    assert np.linalg.norm(nn - np.array([0, 0.867, -0.5])) < 1e-2

    t = np.dot(sigma_g, nn)
    assert np.linalg.norm(t - np.array([0, 11.95, -11.5]))/np.linalg.norm(t) \
            < 1e-2

    sn, tau = calc_normal_and_shear_components(t, nd, ns, nn)
    assert math.fabs((sn - 16.1)/sn) < 1e-2
    assert math.fabs((tau - 3.98)/tau) < 1e-2

def test_2d_strike_slip():
    angles = ct.EulerAngles(120, 0, 90, unit="degrees")
    rot_pg = angles.calc_rotation_matrix()
    sigma_p = np.diag([45, 30, 25])
    sigma_g = ct.rotate_tensor(sigma_p, np.transpose(rot_pg))
    sigma_gc = np.array([[30, -8.66, 0], [-8.66, 40, 0], [0, 0, 30]])
    assert np.linalg.norm(sigma_g - sigma_gc)/np.amax(np.fabs(sigma_g)) < 1e-3

    fault = ct.FaultPlane(60, 90, unit="degrees")
    nd, ns, nn = fault.calc_bases()
    assert np.linalg.norm(nn - np.array([-0.866, 0.5, 0])) < 1e-2

    t = np.dot(sigma_g, nn)
    assert np.linalg.norm(t - np.array([-30.31, 27.5, 0]))/np.linalg.norm(t) \
            < 1e-2

    sn, tau = calc_normal_and_shear_components(t, nd, ns, nn)
    assert math.fabs((sn - 40)/sn) < 1e-2
    assert math.fabs((tau - 8.66)/tau) < 1e-2

def test_3d_normal_faulting():
    angles = ct.EulerAngles(90, 90, 0, unit="degrees")
    rot_pg = angles.calc_rotation_matrix()
    sigma_p = np.diag([5000, 4000, 3000])
    sigma_g = ct.rotate_tensor(sigma_p, np.transpose(rot_pg))
    sigma_gc = np.diag([4000, 3000, 5000])
    assert np.linalg.norm(sigma_g - sigma_gc)/np.amax(np.fabs(sigma_g)) < 1e-3

    # Fault plane 1
    fault = ct.FaultPlane(45, 60, unit="degrees")
    nd, ns, nn = fault.calc_bases()
    assert np.linalg.norm(nn - np.array([-0.612, 0.612, -0.5])) < 1e-2

    t = np.dot(sigma_g, nn)
    assert np.linalg.norm(t - np.array([-2450, 1840, -2500]))/np.linalg.norm(t) \
            < 1e-2

    sn, tau = calc_normal_and_shear_components(t, nd, ns, nn)
    assert math.fabs((sn - 3870)/sn) < 1e-2
    assert math.fabs((tau - math.sqrt(649 ** 2 + 433 ** 2))/tau) < 1e-2

    # Fault plane 2
    fault = ct.FaultPlane(225, 60, unit="degrees")
    nd, ns, nn = fault.calc_bases()
    assert np.linalg.norm(nn - np.array([0.612, -0.612, -0.5])) < 1e-2

    t = np.dot(sigma_g, nn)
    assert np.linalg.norm(t - np.array([2450, -1840, -2500]))/np.linalg.norm(t) \
            < 1e-2

    sn, tau = calc_normal_and_shear_components(t, nd, ns, nn)
    assert math.fabs((sn - 3870)/sn) < 1e-2
    assert math.fabs((tau - math.sqrt(649 ** 2 + 433 ** 2))/tau) < 1e-2

def test_3d_reverse_faulting():
    angles = ct.EulerAngles(150, 0, 0, unit="degrees")
    rot_pg = angles.calc_rotation_matrix()
    sigma_p = np.diag([2400, 1200, 1000])
    sigma_g = ct.rotate_tensor(sigma_p, np.transpose(rot_pg))
    sigma_gc = np.array([[2100, -520, 0], [-520, 1500, 0], [0, 0, 1000]])
    assert np.linalg.norm(sigma_g - sigma_gc)/np.amax(np.fabs(sigma_g)) < 1e-3

    fault = ct.FaultPlane(120, 70, unit="degrees")
    nd, ns, nn = fault.calc_bases()
    assert np.linalg.norm(nn - np.array([-0.814, -0.470, -0.342])) < 1e-2

    t = np.dot(sigma_g, nn)
    assert np.linalg.norm(t - np.array([-1465, -281, -342]))/np.linalg.norm(t) \
            < 1e-2

    sn, tau = calc_normal_and_shear_components(t, nd, ns, nn)
    assert math.fabs((sn - 1441)/sn) < 1e-2
    assert math.fabs((tau - math.sqrt(160 ** 2 + 488 ** 2))/tau) < 1e-2

