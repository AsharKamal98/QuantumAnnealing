import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
import time


def zPauli(n, site, state, act_from_left):
    """
    Applies z Pauli matrix on a given state.

    INPUT:
    ------
    n =     integer(1.2,...), number of qbits
    site =  integer(1,2,...) , site at which Pauli matrix applied
    state = (2**n,2**n) array
    act_from_left = boolean
    """

    #print("state before:", state)
    for index in range(2**n):
        # Finds bit value corresponding to the site
        k = n-site
        bit_value = (index & (1 << k)) >> k

        # If bit value is 1, a sign change should occur
        if bit_value==1:
            if act_from_left:
                state[index] = -state[index]
            else:
                state[:,index] = -state[:,index]
    #print("state after:", state)

    return state

def xPauli(n, site, state, act_from_left):
    print("state before: \n", state)
    transformed_state = np.ones((2**n,2**n)) * 99
    for index in range(2**n):
        # Finds bit value corresponding to the site
        k = n-site
        bit_value = (index & (1 << k)) >> k

        # Toggle bit value
        if act_from_left:
            if bit_value==1:
                new_index = index-2**(n-site) # bit 1 -> 0
                transformed_state[new_index] = state[index]
            else:
                new_index = index+2**(n-site) # bit 0 -> 1
                transformed_state[new_index] = state[index]
        else:
            if bit_value==1:
                new_index+2**(n-site) 
                transformed_state[:,new_index] = state[:,index]
            else:
                new_index-2**(n-site)
                transformed_state[:,new_index] = state[:,index]

    print("state after: \n", transformed_state)

    return transformed_state

def minplusPauli(n, site, state, act_from_left, minus):
    #print("state before: \n", state)
    transformed_state = state.copy()
    for index in range(2**n):
        # Finds bit value corresponding to the site
        k = n-site
        bit_value = (index & (1 << k)) >> k

        # Toggle bit value
        if act_from_left:
            if not minus and bit_value==1:
                new_index = index-2**(n-site) # bit 1 -> 0
                transformed_state[index] -= state[index]
                transformed_state[new_index] += state[index]
            elif minus and bit_value==0:
                new_index = index+2**(n-site) # bit 0 -> 1
                transformed_state[index] -= state[index]
                transformed_state[new_index] += state[index]

        else:
            if not minus and bit_value==1:
                new_index = index-2**(n-site)
                transformed_state[index] -= state[index]
                transformed_state[:,new_index] += state[:,index]
            elif minus and bit_value==0:
                new_index = index+2**(n-site)
                transformed_state[index] -= state[index]
                transformed_state[:,new_index] += state[:,index]


    #print("state after: \n", transformed_state)

    return transformed_state
    

def EigenSystem():
    """ Produces eigenstates/values for n=2 qbits """
    eigenstates = np.identity(2**n) #[np.identity(2**n)[:,i] for i in range(2**n)]
    eigenvalues = [2,0,0,-2]
    return eigenstates, eigenvalues

def Hcommutator(state):
    commutator = zPauli(n, 1, state, act_from_left=True) + zPauli(n, 2, state, act_from_left=True) - \
            zPauli(n, 1, state, act_from_left=False) - zPauli(n, 2, state, act_from_left=False)
    return commutator

def N(eigenvalues, b, a):
    beta = 1/T
    N = 1/(math.exp(beta*(eigenvalues[b]-eigenvalues[a]))-1) if eigenvalues[b] > eigenvalues[a] else 0
    return N

def g(eigenvalues, b, a):
    g = 1 if eigenvalues[b] > eigenvalues[a] else 0
    return g


def LindBladian():
    """ Computes the Lindbladian operator in vectorized form """

    rho_dot = -complex(0,1) * np.kron(np.identity(2**n), H) - np.kron(H.T, np.identity(2**n)
    for a in range(2**n):
        a_state = eigenstates[a]
        for b in range(2**n):
            b_state = eigenstates[b]

            Nba = N(eigenvalues, b, a)
            gba = g(eigenvalues, b, a)
            Nab = N(eigenvalues, a, b)
            gab = g(eigenvalues, a, b)

            res = 0
            for i in range(n):
                res -= (Nba * abs(gba**2) * np.matmul(a_state, minplusPauli(n, i+1, b_state, act_from_left=True, minus=True)) * \
                        np.matmul(b_state, minplusPauli(n, i+1, a_state, act_from_left=True, minus=False)) + \
                        (Nab+1) * abs(gab**2) * np.matmul(b_state, minplusPauli(n, i+1, a_state, act_from_left=True, minus=True)) * \
                        np.matmul(a_state, minplusPauli(n, i+1, b_state, act_from_left=True, minus=False)))

            rho_dot[a] += res * rho[a]
            rho_dot[:,a] += res * rho[:,a]
            rho_dot[b,b] -= res * rho[a,a]

    return None


def Euler(n, rho):
    h = 0.1
    for i in range(300):
        rho_dot = LindBladian(n, rho) 
        rho = rho + h*rho_dot
        if i%10==0:
            print("\nDensity matrix: \n", rho.round(decimals=3))
            print("\nDensity matrix differential: \n", rho_dot.round(decimals=3))
            print("-------------------------------------------------------------")
    
    return None


def CrankNicholsan(n, psi, H):
    dt=0.1
    iterations = 1000
    conv = np.zeros((4,iterations))
    for i in range(iterations):
        #print(np.linalg.det(np.identity(2**n) + (1/2)*complex(0,1)*H*dt))
        #print(np.linalg.inv(np.identity(2**n) + (1/2)*complex(0,1)*H*dt))
        psi_new = np.linalg.inv(np.identity(2**n) + (1/2)*complex(0,1)*H*dt) @ ((np.identity(2**n) - (1/2)*complex(0,1)*H*dt) @ psi)
        conv[:,i] = abs(psi_new - psi)
        psi = psi_new

        if i%(int(iterations/10))==0:
            print("\nstate vector: \n", psi.round(decimals=3))


    for index in range (4):
        plt.plot(np.linspace(0,iterations-1,iterations), conv[index], 'ro', markersize=2)
        plt.show()
    return None


def RungeKutta(n, rho):
    h = 0.1

    for i in range(10):
        k1 = LindBladian(n, rho)
        k2 = LindBladian(n, rho + h*(k1/2))
        k3 = LindBladian(n, rho + h*(k2/2))
        k4 = LindBladian(n, rho + h*k3)
    return None

T = 1
n = 2 # number of qbits
initial_rho = (1/n**0.5) * np.ones((2**n, 2**n)) # Factors wrong!
initial_psi = (1/n**0.5) * np.ones(2**n)
#print(initial_rho)

random_state1 = np.ones((2**n, 2**n))
random_state2 = np.identity(2**n)
random_state3 = np.ones(2**n)

#minplusPauli(n, 1, random_state3, act_from_left=True, minus=True)
#xPauli(n, 1, random_state2, act_from_left=True)
#LindBladian(n, initial_rho)
#Euler(n, initial_rho)
H = Hamiltoneon()
CrankNicholsan(n, initial_psi, H)


#print(rho_initial)
#print(EigenStates(n))

