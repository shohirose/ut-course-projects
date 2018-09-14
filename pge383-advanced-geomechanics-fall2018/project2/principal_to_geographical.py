import geographical_to_principal as gtp
import numpy as np
import math

def rotate_stress_tensor(sigma_p, rot_pg):
    """
    Rotate stress tensor from principal stress to geographical coordinate
    system.

    Parameters
    ---------
    sigma_p : numpy.array
        stress tensor in principal stress coordinate system
    rot_pg : numpy.array
        rotation matrix from geographical to principal stress coordinate system

    Returns
    -------
    sigma_g : numpy.array
        stress tensor in geographical coordinate system
    """

    sigma_g = np.dot(np.dot(np.transpose(rot_pg), sigma_p), rot_pg)
    return sigma_g
