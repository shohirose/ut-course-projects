import numpy as np
import math

def calc_rotation_matrix(alpha, beta, gamma):
    """
    Calculate rotation matrix from geographical to principal stress coordinate
    system.

    Parameters
    ----------
    alpha : double
        Euler angle in radians
    beta : double
        Euler angle in radians
    gamma : double
        Euler angle in radians

    Returns
    -------
    rot_pg : numpy.array
        rotation matrix from geographical to principal stress coordinate system
    """

    c1 = math.cos(alpha)
    s1 = math.sin(alpha)
    c2 = math.cos(beta)
    s2 = math.sin(beta)
    c3 = math.cos(gamma)
    s3 = math.sin(gamma)

    rot_pg = np.array([[c1*c2, s1*c2, -s2],
        [c1*s2*s3 - s1*c3, s1*s2*s3 + c1*c3, c2*s3],
        [c1*s2*c3 + s1*s3, s1*s2*c3 - c1*s3, c2*c3]])
    return rot_pg

def rotate_stress_tensor(sigma_g, rot_pg):
    """
    Rotate stress tensor from geographical to principal stress coordinate system.

    Parameters
    ----------
    sigma_g : numpy.array
        stress tensor in geographical coordinate system.
    rot_pg : numpy.array
        rotation matrix from geographical to principal stress coordinate system.

    Returns
    -------
    sigma_p : numpy.array
        stress tensor in principal stress coordinate system.
    """

    sigma_p = np.dot(np.dot(rot_pg, sigma_g), np.transpose(rot_pg))
    return sigma_p
