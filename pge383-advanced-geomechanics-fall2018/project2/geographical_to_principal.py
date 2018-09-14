import numpy as np
import math

def calc_rotation_matrix(alpha, beta, gamma):
    c1 = math.cos(alpha)
    s1 = math.sin(alpha)
    c2 = math.cos(beta)
    s2 = math.sin(beta)
    c3 = math.cos(gamma)
    s3 = math.sin(gamma)

    return np.array([[c1*c2, s1*c2, -s2],
        [c1*s2*s3 - s1*c3, s1*s2*s3 + c1*c3, c2*s3],
        [c1*s2*c3 + s1*s3, s1*s2*c3 - c1*s3, c2*c3]])

def rotate_stress_tensor(sigma, rot):
    return np.dot(np.dot(rot, sigma), np.transpose(rot))
