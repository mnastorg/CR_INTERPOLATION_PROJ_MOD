import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from random import *
import sys


###################################################################################################################################################
################################################## PROGRAMME PRINCIPALE  ##########################################################################
###################################################################################################################################################

def main_oscillateur():

    print("-------------------------------------------------------------------")
    print("--------------------- DEBUT DU PROGRAMME --------------------------")
    print("-------------------------------------------------------------------")

    #------------------------ DESCRIPTION DES PARAMETRES -----------------------

    #DISCRETISATION TEMPORELLE
    print("------------ Conditions initiales ------------------")
    tmax = 50
    print("tmax = ",tmax)
    t = np.arange(0,tmax+1)

    #CONDITIONS INITIALES DU SYSTEME
    x0 = 0.
    print("x0 = ",x0)
    xdot0 = 0.
    print("xdot0 = ", xdot0)
    Z = [x0,xdot0]
    print("Vecteur des conditions initiales = ", Z)
    print("----------------------------------------------------")

    #PARAMETRES FIXES "UNE BONNE FOIS POUR TOUTE"
    print("-------- Paramètres définis pour le programme ------")
    delta = 0.02
    print("delta = ", delta)
    beta = 5
    print("beta = ", beta)
    omega = 0.5
    print("omega = ", omega)
    print("----------------------------------------------------")

    #PARAMETRES SUR LESQUELS ON JOUE
    alpha = np.linspace(1,3,21)
    print("alpha = ", alpha)
    gamma_1 = 1
    print("gamma_1 = ", gamma_1)
    gamma_2 = 3.5
    print("gamma_2 = ", gamma_2)

    #NOMBRE DE VECTEUR DE LA BASE CONSIDERE
    k = 5
    print("Nombre de vecteur de la base = ", k)

    #sol = solu_oscillateur(Z,t,delta,beta,omega,1,gamma_1)

    #-------------------------- INITIALISATION DES BASES -----------------------
    #PSY_INIT = BASE POUR GAMMA 1
    PSY_INIT = base_gamma(Z,t,delta,beta,omega,alpha,gamma_1)
    #PHI_INIT = BASE POUR GAMMA 2
    PHI_INIT = base_gamma(Z,t,delta,beta,omega,alpha,gamma_2)
    print("La taille de PSY_INIT est = ", np.shape(PSY_INIT))
    print("La taille de PHI_INIT est = ", np.shape(PHI_INIT))

    #------------------------- ON REDUIT ORTHOGONALISE LA BASE -----------------
    print("-----------Quelques calculs sur les dimensions ------")

    PSY_ORTHO = orthonormalisation(PSY_INIT,k)
    PHI_ORTHO = orthonormalisation(PHI_INIT,k)
    #PSY_ORTHO = gram_schmidt(PSY_INIT)
    #PHI_ORTHO = gram_schmidt(PHI_INIT)
    print("La taille de PSY_ORTHO est = ", np.shape(PSY_ORTHO))
    print("La taille de PHI_ORTHO est = ", np.shape(PHI_ORTHO))

    #----------------------- FAIRE INTERPOLATION A t = 0.5 AVEC GEODESIQUE  ------

    INTERPOL = solution_val_lim(PSY_ORTHO,PHI_ORTHO,0.5)
    gamma_mid = (gamma_1 + gamma_2)/2
    print("gamma_mid = ", gamma_mid)

    #----------------------- ON REGARDE LA SOLUTION GROSSIERE MOYENNE ----------

    GROSSIERE = (PHI_INIT + PSY_INIT)/2

    #---------------------- CALCUL D'ERREUR SUR LA SOLUTION --------------------
    print("-------------------Les calculs d'erreur --------------")

    erreur_PSY = erreur_moy_HF(PSY_ORTHO,Z,t,delta,beta,omega,alpha,gamma_1)
    erreur_PHI = erreur_moy_HF(PHI_ORTHO,Z,t,delta,beta,omega,alpha,gamma_2)

    print("L'erreur moyenne sur tous les alphas entre la solution exacte et la solution projetée sur la base réduite PSY est, pour gamma_1 : ", erreur_PSY)
    print("L'erreur moyenne sur tous les alphas entre la solution exacte et la solution projetée sur la base réduite PHI est, pour gamma_2 : ", erreur_PHI)

    erreur_INTERPOL = erreur_moy_HF(INTERPOL,Z,t,delta,beta,omega,alpha,gamma_mid)
    print("L'erreur moyenne sur tous les alphas entre la solution exacte et la solution projetée sur la base INTERPOLE est, pour gamma_mid : ", erreur_INTERPOL)

    erreur_GROSSIERE = erreur_moy_HF(GROSSIERE,Z,t,delta,beta,omega,alpha,gamma_mid)
    print("L'erreur moyenne sur tous les alphas entre la solution exacte et la solution projetée sur la base GROSSIERE est, pour gamma_mid : ", erreur_GROSSIERE)

    print("------------------------------------------------------")

    print("-------------------------------------------------------------------")
    print("--------------------- FIN DU PROGRAMME ----------------------------")
    print("-------------------------------------------------------------------")

    return PSY_INIT, PHI_INIT


###################################################################################################################################################
################################################## FONCTIONS POUR LE MAIN #########################################################################
###################################################################################################################################################


#--------------------------- FONCTION VECTEURS DERIVES -------------------------
def deriv(Z,t,delta,beta,omega,alpha,gamma):
    dX = Z[1]
    dY = -delta*Z[1] - alpha*Z[0] - beta*(Z[0])**3 + gamma*np.cos(omega*t)
    dZ = [dX,dY]
    return dZ
#-------------------------------------------------------------------------------

#----------------------------- SOLUTION DE L'EDO -------------------------------
def solu_oscillateur(Z,t,delta,beta,omega,alpha,gamma):
    #sol est de taille(N,2) ---> ka ===la premiere colonne est la solution
    #et la seconde sa derivée.
    sol = odeint(deriv, Z, t, args = (delta,beta,omega,alpha,gamma))
    return sol
#-------------------------------------------------------------------------------

#------------------------------ FONCTION CREATION BASE GAMMA_I -----------------

def base_gamma(Z,t,delta,beta,omega,alpha,gamma):
    N = np.size(alpha)
    BASE = np.zeros((np.size(t),N))
    for i in range(N):
        SOL = solu_oscillateur(Z,t,delta,beta,omega,alpha[i],gamma)
        BASE[:,i] = SOL[:,0]
    return BASE
#-------------------------------------------------------------------------------

#----------------------------- AFFICHAGE ---------------------------------------
def affichage(t,sol):

    fig = plt.figure(figsize = [7,7])
    ax = fig.add_subplot(111)
    ax.plot(sol[:,0], sol[:,1], 'b', label = 'x(t)')
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Oscillateur de Duffing")

    plt.show()
#-------------------------------------------------------------------------------

#------------------------ FONCTION DE PROJECTION DE V SUR U --------------------
def proj(v,u):

    proj = np.dot(v,u)*u

    return proj
#-------------------------------------------------------------------------------

#------------------------- ORTHONORMALISATION SELON SEUIL ----------------------
def orthonormalisation(MATRIX, nbr_vect):

    N = np.shape(MATRIX)[0]
    k = np.shape(MATRIX)[1]

    ortho = []
    norm = []
    indice = []
    x_gamma = MATRIX[:,0]
    e = x_gamma/np.linalg.norm(x_gamma)
    ortho.append(e)

    for i in range(1,k):
        v = MATRIX[:,i]
        sum = np.zeros(N)

        for i in range(len(ortho)):
            sum += proj(v,ortho[i])

        u = v - sum
        norm.append(np.linalg.norm(u))
        e = u/np.linalg.norm(u)
        ortho.append(e)

    if nbr_vect > len(norm) + 1:
        print("ERREUR, veuillez saisir un nombre moins important de vecteurs de base à conserver")
        sys.exit()
    else:
        m = 0
        while m < nbr_vect - 1:
            indice.append(norm.index(min(norm)))
            norm[norm.index(min(norm))] = max(norm)
            m = m + 1

    ORTHO = np.zeros((N, nbr_vect))
    ORTHO[:, 0] = ortho[0]
    for i in range(len(indice)):
        ORTHO[:, i+1] = ortho[indice[i]+1]

    return ORTHO

#-------------------------------------------------------------------------------

#------------------------- CALCUL ERREUR ENTRE SOL EXACTE ET PROJETEE ----------
def erreur_moy_HF(INIT,Z,t,delta,beta,omega,alpha,gamma):

    PROJ = np.dot(INIT,np.transpose(INIT))
    N = np.size(alpha)
    erreur = 0
    for i in range(N):
        SHF = solu_oscillateur(Z,t,delta,beta,omega,alpha[i],gamma)
        SOL = SHF[:,0]
        erreur += np.linalg.norm(SOL - np.dot(PROJ,SOL))

    return erreur/N

#-------------------------------------------------------------------------------

###################################################################################################################################################
################################################## FONCTIONS POUR L'INTERPOLATION VIA LES GEODESIQUES #############################################
###################################################################################################################################################

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


def gram_schmidt(X):
    Q, R = np.linalg.qr(X, mode = 'reduced')
    return Q
