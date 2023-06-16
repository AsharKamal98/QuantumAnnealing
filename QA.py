import numpy as np
import cmath
import math
#from sympy.physics.quantum import OrthogonalBra as OB
#from sympy.physics.quantum import OrthogonalKet as OK
#from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
#import time


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

            #print("index:", index)
            #print("binary index:", bin(index))
            #print("bit value:", bit_value)
            #print("new index:", new_index)
            #print("-----------------")
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
    



def EigenSystem(n):
    ''' n = number of sites '''
    eigenstates = np.identity(2**n) #[np.identity(2**n)[:,i] for i in range(2**n)]
    eigenvalues = [2,0,0,-2]
    return eigenstates, eigenvalues


def Hcommutator(n, state):
    commutator = zPauli(n, 1, state, act_from_left=True) + zPauli(n, 2, state, act_from_left=True) - \
            zPauli(n, 1, state, act_from_left=False) + zPauli(n, 2, state, act_from_left=False)
    return commutator

def N(eigenvalues, b, a):
    beta = 1/T
    
    if eigenvalues[b] <= eigenvalues[a]:
        N = 0
    else:
        N = 1/(math.exp(beta*(eigenvalues[b]-eigenvalues[a]))-1)
    return N

def g(eigenvalues, b, a):
    if eigenvalues[b] < eigenvalues[a]:
        g = 0
    else:
        g = 1
    return g


def LindBladian(n, rho):
    eigenstates, eigenvalues = EigenSystem(n)

    rho_dot = -complex(0,1) * Hcommutator(n, rho)
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
            rho_dot[a] += res
            rho_dot[:,a] += res
            rho_dot[b,b] -= res

    #print("\nDensity matrix differential: \n", rho_dot.round(decimals=3))
    return rho_dot


def Euler(n, rho):
    h = 0.1
    for i in range(1000):
        rho_dot = LindBladian(n, rho) 
        rho = rho + h*rho_dot
    print("\nDensity matrix: \n", rho_dot.round(decimals=3))
    
    return None


T = 1
n = 2 # number of qbits
initial_rho = (1/n**0.5) * np.ones((2**n, 2**n))
#print(initial_rho)

random_state1 = np.ones((2**n, 2**n))
random_state2 = np.identity(2**n)
random_state3 = np.ones(2**n)

#minplusPauli(n, 1, random_state3, act_from_left=True, minus=True)
#xPauli(n, 1, random_state2, act_from_left=True)
#LindBladian(n, initial_rho)
Euler(n, initial_rho)

#print(rho_initial)
#print(EigenStates(n))

"""
eigenstates = [OK(1,1)*OK(2,1), OK(1,1)*OK(2,0), OK(1,0)*OK(2,1), OK(1,0)*OK(2,0)]
rho_initial = OK(2,0)*OK(1,0)*OB(1,0)*OB(2,0)


#print("Initial rho:", rho_initial)
#print("Eigenstates:", eigenstates)

"""
