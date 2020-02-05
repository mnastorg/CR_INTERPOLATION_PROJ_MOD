import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import *
import sys


################################################################################
#                               PROGRAMME PRINCIPAL
################################################################################

#---------------------------------- FONCTION MAIN ------------------------------

def interpolation_2D(N,k,t):

    PHI,PSY = vect_norm(N,k)

    if verif_proj(PHI,PSY) == 0 :

        print("LES DEUX ESPACES SONT COLINEAIRES, RESTART : NO PAIN NO GAME")
        sys.exit()

    else :

        V, W = vect_correles(PHI,PSY)

        for i in range(k):
            ROT, theta = rotation(V[:,i],W[:,i],t)
            PHI = np.dot(ROT,PHI)
            print("ROT = ", np.shape(ROT))

        gauss, gauss_p_sol = verification(PHI, N)
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
        plt.title("Dans la base interpolée")

        plt.show()
        #theta_tab = np.zeros(np.shape(V)[1])

        #INTERPOL = np.zeros((np.shape(V)[0],np.shape(V)[1]))

        #for i in range(np.shape(V)[1]):
            #theta_tab[i] = np.arccos( np.dot(V[:,i],W[:,i]) / (np.linalg.norm(V[:,i])*np.linalg.norm(W[:,i])) )
            #print(theta_tab[i])
        #print(np.argmin(theta_tab))

            #ROT, theta = rotation(V[:,i],W[:,i],t)
            #theta_tab[i] = theta
            #INTERPOL[:,i] = np.dot(ROT,V[:,i])

    return 0

def verification(SOL, N):

        x = np.linspace(0,1,N)
        y = np.linspace(0,1,N)
        sig0 = uniform(0.005,0.015)

        GAUSS = gaussienne(x, y, N, 0.5, 0.5, sig0)
        gauss = GAUSS.reshape(GAUSS.shape[0]*GAUSS.shape[1])

        P_SOL = np.dot(SOL, np.transpose(SOL))
        gauss_p_sol = np.dot(P_SOL, gauss)


        return gauss, gauss_p_sol

#--------------------- FONCTION ROTATION ENTRE DEUX VECTEURS -------------------

def rotation(G1,G2,t):

    theta = np.arccos(np.dot(G1,G2)/(np.linalg.norm(G1)*np.linalg.norm(G2)))

    w = G2/np.linalg.norm(G2)
    v = (G1 - (np.dot(G1,G2))*(G2/np.linalg.norm(G2)**2)) / np.linalg.norm((G1 - (np.dot(G1,G2))*(G2/np.linalg.norm(G2)**2)))

    M = np.zeros((2,np.size(v)))
    M[0,:] = v
    M[1,:] = w

    P = np.dot(np.transpose(M),M)

    O = mat_rotation(theta,t)

    I = np.eye(np.shape(P)[0])

    ROT = (I-P) + np.dot(np.dot(np.transpose(M),O),M)

    return ROT, theta

#-------------------------------------------------------------------------------
#--------------------- MATRICE DE ROTATION -------------------------------------

def mat_rotation(theta,t):

    MAT = [[np.cos(theta*t), -np.sin(theta*t)],
            [np.sin(theta*t), np.cos(theta*t)] ]

    return MAT

#-------------------------------------------------------------------------------
#--------------------- FONCTION VECTEURS LES MIEUX CORRELES --------------------
def vect_correles(PHI,PSY):

    Q = np.dot(np.transpose(PSY),PHI)
    MAT_X = np.dot(np.transpose(Q),Q)
    MAT_Y = np.dot(Q,np.transpose(Q))

    VALP_X,VECP_X = np.linalg.eig(MAT_X)
    print(VALP_X)
    VALP_Y,VECP_Y = np.linalg.eig(MAT_Y)
    print(VALP_Y)

    V = np.zeros((np.shape(PHI)[0],np.size(VALP_X)))
    W = np.zeros((np.shape(PSY)[0],np.size(VALP_Y)))

    for i in range(np.size(VALP_X)):
        print("VALP_X = ", VALP_X)
        index = np.argmax(VALP_X)
        print("INDEX = ", index)
        print("VECP_X_I = ", VECP_X[:,index])
        V[:,i] = np.dot(PHI,VECP_X[:,index])
        W[:,i] = np.dot(PSY,VECP_Y[:,index])
        VALP_X[index] = -100

    return V, W

#-------------------------------------------------------------------------------
#------------------------ INITIALISATION ET NORMALISATION GAUSSIENNE -----------

def vect_norm(N,k):

    #DEFINITION DES PARAMETRES
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    x0,y0 = 0.45, 0.5
    x1,y1 = 0.55, 0.5

    #MATRICE POUR STOCKER LES GAUSSIENNES
    PHI = np.zeros((N*N, k))
    PSY = np.zeros((N*N, k))

    for i in range(k):
        sig0 = uniform(0.005,0.015)
        sig1 = uniform(0.005,0.015)
        G1 = gaussienne(x,y,N,x0,y0,sig0)
        G2 = gaussienne(x,y,N,x1,y1,sig1)
        PHI[:,i] = G1.reshape(G1.shape[0]*G1.shape[1])
        PSY[:,i] = G2.reshape(G2.shape[0]*G2.shape[1])

    PHI_NORM = gram_schmidt(PHI)
    PSY_NORM = gram_schmidt(PSY)

    return PHI_NORM, PSY_NORM

#-------------------------------------------------------------------------------
#---------------------- VERIFICATION DIFFERENCIATION SOUS ESPACES --------------

def verif_proj(PHI,PSY):

    N  = np.shape(PHI)[0]
    #DEFINITION DES PARAMETRES
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    V = gaussienne(x,y,N,0.5,0.5,0.01)
    #PROJECTEURS
    P_PHI = np.dot(PHI,np.transpose(PHI))
    P_PSY = np.dot(PSY,np.transpose(PSY))
    #TESTS
    RESULT = np.dot((P_PHI-P_PSY),V)
    if RESULT.all() == 0 :
        return 0
    else:
        return 1

#-------------------------------------------------------------------------------
#---------------------- FONCTION CREATION GAUSSIENNE ---------------------------

def gaussienne(x,y,N,x0,y0,sig1):
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
