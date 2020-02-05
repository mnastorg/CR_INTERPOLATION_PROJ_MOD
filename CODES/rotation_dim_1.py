import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotation(G1,G2,t):

    G1 = G1.reshape(G1.shape[0]*G1.shape[1])
    G2 = G2.reshape(G2.shape[0]*G2.shape[1])

    theta = np.arccos(np.dot(G1,G2)/(np.linalg.norm(G1)*np.linalg.norm(G2)))

    w = G2/np.linalg.norm(G2)
    v = (G1 - (np.dot(G1,w))*w)/np.linalg.norm((G1 - (np.dot(G1,w))*w))

    M = np.zeros((2,np.size(v)))

    M[0,:] = v
    M[1,:] = w

    P = np.dot(np.transpose(M),M)

    O = mat_rotation(theta,t)

    I = np.eye(np.shape(P)[0])

    ROT = (I-P) + np.dot(np.dot(np.transpose(M),O),M)

    return ROT


def mat_rotation(theta,t):

    MAT = [ [np.cos(theta*t),-np.sin(theta*t)],
            [np.sin(theta*t),np.cos(theta*t)] ]

    return MAT

def gaussienne(x,y,N,x0,y0,x1,y1,sig1,sig2):

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    G1 = np.zeros((N+1,N+1))
    G2 = np.zeros((N+1,N+1))

    for i in range(N+1):
        for j in range(N+1):
            G1[i,j] = np.exp(-(((x[i]-x0)**2)/(2*sig1) + ((y[j]-y0)**2)/(2*sig1)))
            G2[i,j] = np.exp(-(((x[i]-x1)**2)/(2*sig2) + ((y[j]-y1)**2)/(2*sig2)))

    return G1,G2

def affichage(FILE):

    INTERPOL = np.loadtxt(FILE)

    x = np.linspace(0,1,np.shape(INTERPOL)[0])
    y = np.linspace(0,1,np.shape(INTERPOL)[0])

    fig = plt.figure(figsize = [10,10])
    X,Y = np.meshgrid(x,y)
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(X,Y,INTERPOL, cmap = 'hot')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("INTERPOLATION")

    #fig2 = plt.figure(figsize = [10,10])
    #X,Y = np.meshgrid(x,y)
    #ax2 = fig2.add_subplot(111, projection = '3d')
    #ax2.plot_surface(X,Y,INTERPOL, cmap = 'hot')
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.title("INTERPOLATION")

    plt.show()

def main(N):

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    G1, G2 = gaussienne(x,y,N,0.4,0.4,0.6,0.6,0.01,0.01)


    GR1 = G1.reshape(G1.shape[0]*G1.shape[1],1)
    GR2 = G2.reshape(G2.shape[0]*G2.shape[1],1)

    for t in np.arange(0,1.1,0.1):
        t = round(t,3)
        ROT = rotation(G1,G2,t)
        GG1 = np.dot(ROT,GR1)
        GG1 = GG1.reshape(G1.shape[0],G1.shape[1])
        np.savetxt("output1/Interpol_{}.txt".format(t),GG1)

    for t in np.arange(0,2.1,0.1):
        t = round(t,3)
        affichage("output1/Interpol_{}.txt".format(t))
