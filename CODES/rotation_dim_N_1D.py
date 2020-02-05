import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import *
import sys

def interpolation_1D(N,k,t):

    PHI,PSY = vect_norm(N,k)

    if verif_proj(PHI,PSY) == 0 :

        print("LES DEUX ESPACES SONT COLINEAIRES, RESTART : NO PAIN NO GAME")
        sys.exit()

    else :

        V, W = vect_correles(PHI,PSY)

        theta_tab = np.zeros(np.shape(V)[1])

        #for i in range(np.shape(V)[1]):
        #    theta_tab[i] = np.arccos( np.dot(V[:,i],W[:,i]) / (np.linalg.norm(V[:,i])*np.linalg.norm(W[:,i])) )
        #    print(theta_tab[i])
        #print(np.argmin(theta_tab))

        for i in range(k):
            ROT, theta = rotation(V[:,i],W[:,i],t)
            PHI = np.dot(ROT,PHI)
            print("ROT = ", np.shape(ROT))

        GAUSS, GAUSS_PROJ = verification(PHI, N)
        ERREUR = np.linalg.norm(GAUSS - GAUSS_PROJ)

        print("Erreur : ", ERREUR)

        x = np.linspace(0, 1, N)
        fig = plt.figure(figsize = [10,10])
        ax = fig.add_subplot(111)
        ax.plot(x,GAUSS)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("INITIAL")

        fig = plt.figure(figsize = [10,10])
        ax = fig.add_subplot(111)
        ax.plot(x,GAUSS_PROJ)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Dans la base interpolée pour k = {}".format(k))

        plt.show()

        theta_tab = np.zeros(np.shape(V)[1])

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
        sig0 = uniform(0.005,0.015)

        GAUSS = gaussienne(x, N, 0.5, sig0)

        P_SOL = np.dot(SOL, np.transpose(SOL))
        GAUSS_PROJ = np.dot(P_SOL, GAUSS)

        return GAUSS, GAUSS_PROJ


def evolution_erreur():
    x = np.arange(5, 55, 5)
    ERREUR = np.zeros(int(50/5))
    for i in range(10):
        ERREUR[i] = interpolation_1D(50,x[i],0.5)

    x = np.arange(5, 55, 5)
    fig = plt.figure(figsize = [10,10])
    ax = fig.add_subplot(111)
    ax.plot(x,ERREUR)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Erreur en fonction dun nombre de vecteurs")


def gaussienne(x,N,x0,sig1):
    G = np.zeros(N)
    for i in range(N):
        G[i] = np.exp(-(((x[i]-x0)**2)/(2*sig1)))

    return G

def gram_schmidt(X):
    Q, R = np.linalg.qr(X, mode = 'reduced')

    return Q

def vect_norm(N,k):

    #DEFINITION DES PARAMETRES
    x = np.linspace(0, 1, N)
    x0 = 0.45
    x1 = 0.55

    #MATRICE POUR STOCKER LES GAUSSIENNES
    PHI = np.zeros((N, k))
    PSY = np.zeros((N, k))

    for i in range(k):
        sig0 = uniform(0.005,0.015)
        sig1 = uniform(0.005,0.015)
        G1 = gaussienne(x,N,x0,sig0)
        G2 = gaussienne(x,N,x1,sig1)
        PHI[:,i] = G1
        PSY[:,i] = G2

    PHI_NORM = gram_schmidt(PHI)
    PSY_NORM = gram_schmidt(PSY)

    return PHI_NORM, PSY_NORM


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
        index1 = np.argmax(VALP_X)
        index2 = np.argmax(VALP_Y)
        print("INDEX_X = ", index1)
        print("INDEX_Y = ", index2)
        print("VECP_X_I = ", VECP_X[:,index1])
        print("VECP_Y_I = ", VECP_Y[:,index2])
        V[:,i] = np.dot(PHI,VECP_X[:,index1])
        W[:,i] = np.dot(PSY,VECP_Y[:,index2])
        VALP_X[index1] = -100
        VALP_Y[index2] = -100

    return V, W


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


def mat_rotation(theta,t):

    MAT = [[np.cos(theta*t), -np.sin(theta*t)],
            [np.sin(theta*t), np.cos(theta*t)] ]

    return MAT

def verif_proj(PHI,PSY):

    N  = np.shape(PHI)[0]
    #DEFINITION DES PARAMETRES
    x = np.linspace(0,1,N)
    V = gaussienne(x,N,0.5,0.01)
    #PROJECTEURS
    P_PHI = np.dot(PHI,np.transpose(PHI))
    P_PSY = np.dot(PSY,np.transpose(PSY))
    #TESTS
    RESULT = np.dot((P_PHI-P_PSY),V)
    if RESULT.all() == 0 :
        return 0
    else:
        return 1
