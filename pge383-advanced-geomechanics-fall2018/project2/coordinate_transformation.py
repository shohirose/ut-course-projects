import numpy as np
import math

class EulerAngles():
    """
    Class for Euler angles.
    """

    def __init__(self, alpha, beta, gamma, unit="radians"):
        self.set(alpha, beta, gamma, unit)

    def set(self, alpha, beta, gamma, unit="radians"):
        """
        Set Euler angles.
        """

        assert unit == "radians" or unit == "degrees"

        if unit == "radians":
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
        else:
            self.alpha = math.radians(alpha)
            self.beta = math.radians(beta)
            self.gamma = math.radians(gamma)

    def calc_rotation_matrix(self):
        """
        Calculate rotation matrix from old to new coordinate system.
        """

        # c1 = math.cos(self.alpha)
        # s1 = math.sin(self.alpha)
        # c2 = math.cos(self.beta)
        # s2 = math.sin(self.beta)
        # c3 = math.cos(self.gamma)
        # s3 = math.sin(self.gamma)
        #
        # return np.array([[c1*c2, s1*c2, -s2],
        #             [c1*s2*s3 - s1*c3, s1*s2*s3 + c1*c3, c2*s3],
        #             [c1*s2*c3 + s1*s3, s1*s2*c3 - c1*s3, c2*c3]])

        e1, e2, e3 = self.calc_bases()
        return np.array([e1, e2, e3])

    def calc_bases(self):
        """
        Calculate bases of the new coordinate system

        Returns
        -------
        e1 : numpy.array
            Unit vector in the new x1 direction
        e2 : numpy.array
            Unit vector in the new x2 direction
        e3 : numpy.array
            Unit vector in the new x3 direction
        """

        c1 = math.cos(self.alpha)
        s1 = math.sin(self.alpha)
        c2 = math.cos(self.beta)
        s2 = math.sin(self.beta)
        c3 = math.cos(self.gamma)
        s3 = math.sin(self.gamma)

        e1 = np.array([c1*c2, s1*c2, -s2])
        e2 = np.array([c1*s2*s3 - s1*c3, s1*s2*s3 + c1*c3, c2*s3])
        e3 = np.array([c1*s2*c3 + s1*s3, s1*s2*c3 - c1*s3, c2*c3])

        return e1, e2, e3

class FaultPlane():
    """
    Class for fault plane represented by dip and strike angles.
    (d-s-n coordinate system)
    """

    def __init__(self, strike, dip, unit="radians"):
        self.set(strike, dip, unit)

    def set(self, strike, dip, unit="radians"):
        """
        Set strike and dip angles.
        """

        assert unit == "radians" or unit == "degrees"

        if unit == "radians":
            self.strike = strike
            self.dip = dip
        else:
            self.strike = math.radians(strike)
            self.dip = math.radians(dip)

    def calc_rotation_matrix(self):
        """
        Calculate rotation matrix from old to new coordinate system.
        """

        c1 = math.cos(self.strike)
        s1 = math.sin(self.strike)
        c2 = math.cos(self.dip)
        s2 = math.sin(self.dip)

        return np.array([[-s1*c2, c1*c2, s2]],
                         [c1, s1, 0],
                         [-s1*s2, c1*s2, -c2])

    def calc_bases(self):
        """
        Calculate bases of the new coordinate system

        Returns
        -------
        nd : np.array
            Unit vector in the dip direction
        ns : np.array
            Unit vector in the strike direction
        nn : np.array
            Unit vector in the fault normal direction
        """

        c1 = math.cos(self.strike)
        s1 = math.sin(self.strike)
        c2 = math.cos(self.dip)
        s2 = math.sin(self.dip)

        nd = np.array([-s1*c2, c1*c2, s2])
        ns = np.array([c1, s1, 0])
        nn = np.array([-s1*s2, c1*s2, -c2])

        return nd, ns, nn


def rotate_tensor(tensor, rot):
    """
    Transform a given tensor from the old to new coordinate system using the
    given rotation matrix.
    The rotation matrix represents rotation from old to new coordinate system.

    Parameters
    ----------
    tensor : numpy.array
        Tensor in the old coordinate system.
    rot : numpy.array
        Rotation matrix from old to new coordinate system.

    Returns
    -------
    tensor_new : numpy.array
        Tensor in the new coordinate system.
    """

    tensor_new = np.dot(np.dot(rot, tensor), np.transpose(rot))
    return tensor_new

