import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import *
import sys


def main_geodesique(N, k, t):

    PHI, PSY = vect_norm(N, k)

    SOL = solution_val_lim(PHI, PSY, t)

    gauss, gauss_p_sol = verification(SOL, N)

    ERREUR = np.linalg.norm(gauss - gauss_p_sol)
    print("Erreur :", ERREUR)

    GAUSS = gauss.reshape((N, N))
    GAUSS_P_SOL = gauss_p_sol.reshape((N, N))

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    fig = plt.figure(figsize = [10,10])
    X,Y = np.meshgrid(x,y)
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(X,Y,GAUSS, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("INITIAL")

    fig = plt.figure(figsize = [10,10])
    X,Y = np.meshgrid(x,y)
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(X,Y,GAUSS_P_SOL, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Dans la base interpolée pour {} vecteurs".format(k))

    plt.show()

    return ERREUR

def verification(SOL, N):

        x = np.linspace(0,1,N)
        y = np.linspace(0,1,N)
        sig0 = uniform(0.005,0.015)

        GAUSS = gaussienne(x, y, N, 0.5, 0.5, sig0)
        gauss = GAUSS.reshape(GAUSS.shape[0]*GAUSS.shape[1])

        P_SOL = np.dot(SOL, np.transpose(SOL))
        gauss_p_sol = np.dot(P_SOL, gauss)


        return gauss, gauss_p_sol

def evolution_erreur_geo():
    x = np.arange(5, 400, 25)
    ERREUR = np.zeros(int(400/25))
    for i in range(16):
        ERREUR[i] = main_geodesique(20,x[i],0.5)

    fig = plt.figure(figsize = [10,10])
    ax = fig.add_subplot(111)
    ax.plot(x,ERREUR)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Erreur/ nombre vecteurs")


def solution_val_lim(PHI, PSY, t):

    U, SIGMA, V_T = svd(PHI, PSY)

    V = np.transpose(V_T)
    N = np.shape(SIGMA)[0]

    COS_T_SIGMA = np.zeros((N,N))
    SIN_T_SIGMA = np.zeros((N,N))

    for i in range(N):
        COS_T_SIGMA[i, i] = np.cos(t*SIGMA[i,i])
        SIN_T_SIGMA[i, i] = np.sin(t*SIGMA[i,i])

    #SOLUTION DE LA FORME : GAMMA(t) = GAMMA(0).V.cos(SIGMA*t) + U.sin(SIGMA*t)
    P1 = np.dot(np.dot (PHI,V), COS_T_SIGMA)
    P2 = np.dot(U, SIN_T_SIGMA)

    GAMMA_t = P1 + P2

    return GAMMA_t


def svd(PHI, PSY):

    PHI_PHI_T = np.dot(PHI, np.transpose(PHI))
    N = np.shape(PHI_PHI_T)[0]
    I = np.eye(N)
    PHI_T_PSY = np.dot(np.transpose(PHI) , PSY)
    INV_PHI_T_PSY = np.linalg.inv(PHI_T_PSY)

    #INITIALISATION DE LA MATRICE SUR LEQUEL ON VA FAIRE LA SVD
    POUR_SVD = np.dot( np.dot( I - PHI_PHI_T , PSY ), INV_PHI_T_PSY )

    #SVD (ATTENTION : LE SIGMA SORTANT DE np.linalg.svd EST UN VECTEUR (k,))
    U, tnSIGMA, V_T = np.linalg.svd(POUR_SVD, full_matrices = False)

    SIGMA = np.arctan(tnSIGMA)

    #LE VECTEUR SIGMA TRANSFORME EN MATRICE (k,k)
    SIGMA = np.diag(SIGMA)

    return U, SIGMA, V_T


def vect_norm(N, k):

    #DEFINITION DES PARAMETRES
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    x0,y0 = 0.45, 0.45
    x1,y1 = 0.55, 0.55

    #MATRICE POUR STOCKER LES GAUSSIENNES
    PHI = np.zeros((N*N,k))
    PSY = np.zeros((N*N,k))
    PHI_NORM = np.zeros((N,k))
    PSY_NORM = np.zeros((N,k))

    for i in range(k):
        sig0 = uniform(0.005,0.015)
        sig1 = uniform(0.005,0.015)
        G1 = gaussienne(x, y, N,x0,y0,sig0)
        G2 = gaussienne(x, y, N,x1,y1,sig1)
        PHI[:,i] = G1.reshape(G1.shape[0]*G1.shape[1])
        PSY[:,i] = G2.reshape(G2.shape[0]*G2.shape[1])

    PHI_NORM = gram_schmidt(PHI)
    PSY_NORM = gram_schmidt(PSY)

    return PHI_NORM, PSY_NORM


#-------------------------------------------------------------------------------
#---------------------- FONCTION CREATION GAUSSIENNE ---------------------------

def gaussienne(x, y, N, x0, y0, sig1):
    G1 = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            G1[i,j] = np.exp(-(((x[i]-x0)**2)/(2*sig1) + ((y[j]-y0)**2)/(2*sig1)))

    return G1

#-------------------------------------------------------------------------------
#-------------------- FONCTION NORMALISATION MATRICE ---------------------------

def gram_schmidt(X):
    Q, R = np.linalg.qr(X, mode = 'reduced')
    return Q
