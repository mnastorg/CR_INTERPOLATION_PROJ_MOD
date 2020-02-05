import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import *
import sys


def solution_val_init(GAMMA, GAMMA_PT, t):

    U, SIGMA, V_T = svd(GAMMA_PT)
    V = np.transpose(V_T)
    GAMMA_t = np.dot(np.dot(GAMMA, V), np.cos(t*SIGMA)) + np.dot(U, np.sin(t*SIGMA))

    return GAMMA_t

def svd(GAMMA_PT):

    U, SIGMA, V_T = np.linalg.svd(GAMMA_PT, full_matrices=False)
    SIGMA = np.diag(SIGMA)

    return U, SIGMA, V_T
