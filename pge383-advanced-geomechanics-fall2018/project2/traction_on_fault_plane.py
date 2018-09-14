import numpy as np
import math

def calc_dsn_coord_system_basis_vectors(strike, dip):
    """
    Calculate basis vectors of a d-s-n (dip-strike-normal) coorcinate system.

    Parameters
    ----------
    strike : double
        strike angle of a fault in radians
    dip : double
        dip angle of a fault in radians

    Returns
    -------
    nd : numpy.array
        the unit vector in the dip direction
    ns : numpy.array
        the unit vector in the strike direction
    nn : numpy.array
        the unit vector in the fault normal direction
    """

    c1 = math.cos(strike)
    s1 = math.sin(strike)
    c2 = math.cos(dip)
    s2 = math.sin(dip)

    nn = np.array([-s1*s2, c1*s2, -c2])
    ns = np.array([c1, s1, 0])
    nd = np.array([-s1*c2, c1*c2, s2])

    return nd, ns, nn

def calc_normal_and_shear_stress(sigma_g, strike, dip):
    """
    Calculate normal and shear components of traction acting on a fualt plane.

    Parameters
    ----------
    sigma_g : numpy.array
        stress tensor in geographical coordinate system
    strike : double
        strike angle of a fault in radians
    dip : double
        dip angle of a fault in radians

    Returns
    -------
    sn : double
        the normal component of the traction acting on the given plane
    tau : double
        the shear component of the traction acting on the given plane
    """

    nd, ns, nn = calc_dsn_coord_system_basis_vectors(strike, dip)
    t = np.dot(sigma_g, nn)
    sn = np.dot(t, nn)
    td = np.dot(t, nd)
    ts = np.dot(t, ns)
    print(t)
    print(sn)
    print(td)
    print(ts)
    tau = math.sqrt(td ** 2 + ts ** 2)
    return sn, tau
