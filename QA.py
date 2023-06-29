import numpy as np
from numpy import random
import cmath
import math
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time


def zPauli(site, state, act_from_left=True):
    """
    Applies sigma_z on a given state.

    INPUT:
    ------
    site =  integer(1,2,...). The spin/qbit at which Pauli matrix applied
    state = array of size (2**n, 2**n). Single arrays of size 2**n can also be given. 
    act_from_left = boolean. If True, acts on ket.
    """

    #print("state before:\n", state)
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
    #print("state after:\n", state)

    return state


def xPauli(site, state, act_from_left):
    """
    Applies sigma_x on a given state. For information on input, see above. 
    """

    #print("state before: \n", state)
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
                new_index = index-2**(n-site) 
                transformed_state[:,new_index] = state[:,index]
            else:
                new_index = index+2**(n-site)
                transformed_state[:,new_index] = state[:,index]

    #print("state after: \n", transformed_state)

    return transformed_state


def minplusPauli(site, state, act_from_left=True, minus=None):
    """
    Applies Reasing/lowering operator sigma_(plus/minus) on a given state.

    INPUT:
    ------
    site =          site =  integer(1,2,...). The spin/qbit at which Pauli matrix applied 
    state =         array(s) of size 2**n. If multiple arrays are provided, place them into a matrix.
    act_from_left = boolean. If True, Pali matrix acts on on ket.
    minus =         boolean. If True, lowering operator applied, else raising operator. 
    """

    print("state before: \n", state)
    #transformed_state = state.copy()
    transformed_state = np.zeros((2**n,2**n))
    for index in range(2**n):
        # Finds bit value corresponding to the site
        k = n-site
        bit_value = (index & (1 << k)) >> k

        # Toggle bit value
        if act_from_left:
            if not minus and bit_value==1:
                new_index = index-2**(n-site) # bit 1 -> 0
                #transformed_state[index] -= state[index]
                transformed_state[new_index] += state[index]
            #elif not minus and bit_value==0:
            #    transformed_state[index] -= state[index]
            elif minus and bit_value==0:
                new_index = index+2**(n-site) # bit 0 -> 1
                #transformed_state[index] -= state[index]
                transformed_state[new_index] += state[index]
            #elif minus and bit_value==1:
            #    transformed_state[index] -= state[index]

        else:   # Fix!
            if not minus and bit_value==0:
                new_index = index+2**(n-site)
                transformed_state[:,new_index] += state[:,index]
            elif minus and bit_value==1:
                new_index = index-2**(n-site)
                transformed_state[:,new_index] += state[:,index]


    print("state after: \n", transformed_state)

    return transformed_state
    
def RemoveGroundState(site, state, act_from_left=True):
    #print("phi before: \n", state)
    transformed_state = state.copy()
    for index in range(2**n):
        # Finds bit value corresponding to the site
        k = n-site
        bit_value = (index & (1 << k)) >> k

        # Remove if in ground state at given site
        if act_from_left:
            if bit_value==1:
                transformed_state[index] -= state[index]

        else:   # Fix!
            print("do something")

    #print("state after: \n", transformed_state)

    return transformed_state


def EigenSystem():
    """ Produces eigenstates/values for n=2 qbits """
    eigenstates = [np.identity(2**n)[:,i] for i in range(2**n)]
    eigenvalues = [2,0,0,-2]
    return eigenstates, eigenvalues

def Hamiltoneon():
    """ Constructs Hamiltoneon specificially for n=2 qbits case """
    sz = np.array([[1,0],[0,-1]])
    I = np.identity(n)
    H = np.kron(sz, I) + np.kron(I, sz)
    return H

def Hcommutator(rho):
    commutator = zPauli(n, 1, rho, act_from_left=True) + zPauli(n, 2, rho, act_from_left=True) - \
            zPauli(n, 1, rho, act_from_left=False) - zPauli(n, 2, rho, act_from_left=False)
    return commutator

def N(eigenvalues, b, a):
    beta = 1/T
    N = 1/(math.exp(beta*(eigenvalues[b]-eigenvalues[a]))-1) if eigenvalues[b] > eigenvalues[a] else 0
    return N

def g(eigenvalues, b, a):
    g = lam_sq if eigenvalues[b] > eigenvalues[a] else 0
    return g

def deltaMatrix(index):
    """ Creates a matrix of zeros, except for a 1 in the diagonal element [index,index].
        This is used later to pick out certain elements of a matrix (rho). """
    delta = np.zeros((2**n,2**n))
    delta[index,index] = 1
    return delta

def PermMatrix(a, b):
    """ Creates an identity matrix where the a and b rows are swapped. 
        This is used later to move elements of a matrix (rho) along the diagonal. """
    E = np.identity(2**n)
    temp = E[a].copy()
    E[a] = E[b]
    E[b] = temp
    return E


def LindBladian(operator=True, rho=None):
    """ 
    Computes the Lindbladian operator in vectorized form.

    INPUT: 
    ------
    operator =  boolean. If True, does not apply the rho onto the Linbladian operator.
    rho =       array (vectorized matrix) of size 4**n. Only used if operator=False.

    RETURNS
    ------
    If operator = True, returns vectorized Linbladian operator as a (4**n,4**n) array.
    If operator = False, returns vectorized rho differential as a 4**n array.
    """

    H = Hamiltoneon()
    eigenstates,eigenvalues = EigenSystem()

    if operator:
        rho_dot = -complex(0,1) * (np.kron(np.identity(2**n), H) - np.kron(H.T, np.identity(2**n)))
    else:   # Do not need to vectorize here (yet)
        rho_dot = -complex(0,1) * (np.kron(np.identity(2**n), H) - np.kron(H.T, np.identity(2**n))) @ rho

    for a in range(2**n):
        a_state = eigenstates[a]    # Careful! Treating bras and kets the same here. act_from_right will not work.
        for b in range(2**n):
            b_state = eigenstates[b]

            Nba = N(eigenvalues, b, a) # This part can be made faster.
            gba = g(eigenvalues, b, a)
            Nab = N(eigenvalues, a, b)
            gab = g(eigenvalues, a, b)

            pre_factor = 0
            for i in range(n):
                pre_factor -= Nba * abs(gba)**2 * np.matmul(a_state, minplusPauli(i+1, b_state, act_from_left=True, minus=True)) * \
                        np.matmul(b_state, minplusPauli(i+1, a_state, act_from_left=True, minus=False)) + \
                        (Nab+1) * abs(gab)**2 * np.matmul(b_state, minplusPauli(i+1, a_state, act_from_left=True, minus=True)) * \
                        np.matmul(a_state, minplusPauli(i+1, b_state, act_from_left=True, minus=False))

            if operator:
                rho_dot += np.kron(np.identity(2**n), deltaMatrix(a))   # Picks out row a when multiplied with rho.
                rho_dot += np.kron(deltaMatrix(a), np.identity(2**n))   # Picks out column a when multiplied with rho.

                E = PermMatrix(a,b)
                delta = deltaMatrix(a)
                rho_dot -= (np.kron(np.identity(2**n), E) @ np.kron(E, np.identity(2**n))) @ (np.kron(np.identity(2**n), deltaMatrix(a)) @ \
                        np.kron(deltaMatrix(a), np.identity(2**n)))     # Picks out a'th diagonal element and moves it to the b'th diagonal when multiplied with rho.


            else:
                rho_dot[a::2**n] += pre_factor * rho[a::2**n]                       # Row a of rho matrix
                rho_dot[a*2**n:(a+1)*2**n] += pre_factor * rho[a*2**n:(a+1)*2**n]   # Column a of rho matrix
                rho_dot[b*2**n + b] -= pre_factor * rho[a*2**n + a]                 # a'th diagonal element of rho matrix

    return rho_dot


def CrankNicholsan(rho):
    dt=1
    iterations = 200000
    conv = np.zeros((4**n,iterations))

    print("initial rho:\n", rho)

    # Without time dependence, Lindbladian stays the same at each iteration/time step.
    rho_dot_operator = LindBladian(operator=True)

    for i in range(iterations):
        # RHS of Crank Nicholsan
        #rho_dot = LindBladian(operator=False, rho=rho)
        #rho_prime_old = rho + (1/2)*rho_dot*dt
        rho_prime_old = (1 - (1/2)*complex(0,1)*rho_dot_operator*dt) @ rho

        # LHS of Crank Nicholsan
        rho_new = np.linalg.solve((1 + (1/2)*complex(0,1)*rho_dot_operator*dt), rho_prime_old)
        #rho_new = np.linalg.solve((1 + (1/2)*rho_dot_operator*dt), rho_prime_old)
        
        #print(np.sum((1 + (1/2)*rho_dot_operator*dt)/(1 - (1/2)*rho_dot_operator*dt)))


        conv[:,i] = np.absolute(rho_new-rho) #np.absolute(rho_dot)
        rho = rho_new.copy()

        #if i%(int(iterations/5))==0:
            #print("\nstate vector: \n", rho.round(decimals=3))
    print("\nfinal rho: \n", rho.round(decimals=3))

    for index in range (4**n):
        if index < 1:
            plt.plot(np.linspace(0,iterations-1,iterations), conv[index], 'ro', markersize=2)
            plt.show()
    return None

def DirectMethod(rho):
    """ Computes \rho_{n+1} = exp(L*dt) \rho_n iteratively """

    dt = 0.0001
    iterations=20000

    print("initial rho:\n", rho)

    lindbladian = LindBladian(operator=True)
    conv = np.zeros((4**n,iterations))

    for i in range(iterations):
        rho_new = expm(lindbladian*dt) @ rho
        conv[:,i] = np.absolute(rho_new-rho) #np.absolute(rho_dot)
        rho = rho_new.copy()

    print("\nfinal rho: \n", rho.round(decimals=3))

    # Below plots the difference between \rho_{n+1} and \rho_n for each element.
    # Comment out if you do not want to plot all graphs, or reduce number in if statement. 
    for index in range (4**n):
        if index < 2:
            plt.plot(np.linspace(0,iterations-1,iterations), conv[index], 'ro', markersize=2)
            plt.show()

    return None


def MCWF(phi):
    print("initial phi\n", phi)

    dt = 0.0001
    iterations = 80000
    conv = np.zeros((2**n,iterations))

    for i in range(iterations):
        delta_p_list = []
        for j in range(n):
            delta_p_list.append((phi @ RemoveGroundState(j+1, phi)) * dt)
        delta_p = np.sum(delta_p_list)
        epsilon = random.rand()

        if delta_p > 0.5:
            print("Warning! delta_p is getting large, must be much smaller than 1. Current value:", delta_p)

        if epsilon > delta_p:
            #phi_1 = phi - complex(0,1) * (zPauli(1,phi) + zPauli(2,phi)) * dt 
            phi_1 = phi

            for j in range(n):
                phi_1 -= (1/2) * RemoveGroundState(j+1, phi) * dt
            phi_new = phi_1/(1-delta_p)**0.5

        else:
            print("spontaneous emission!")
            max_prob = max(delta_p_list)
            index = np.argmax(delta_p_list)
            phi_new = minplusPauli(index+1, phi, minus=True)/(max_prob/dt)**0.5

        conv[:,i] = np.absolute(phi_new-phi)


        phi = phi_new

        phi_len = np.linalg.norm(phi)
        if phi_len > 1.5 or phi_len < 0.5:
            print("phi not normalized properly. Its norm is\n", phi_len)
    

    for index in range (2**n):
        plt.plot(np.linspace(0,iterations-1,iterations), conv[index], 'ro', markersize=2)
        plt.show()
    #print("\nconvv\n", convv)


    print("new phi\n", phi.round(decimals=3))
    print("phi length:", phi_len)

    return None

def TestFcn(rho):
    """ Ignore """

    print("initial rho:\n", rho)
    
    a = 1
    b = 2

    E = PermMatrix(a,b)
    delta = deltaMatrix(a)

    #transf = (np.kron(np.identity(2**n), E) @ np.kron(E, np.identity(2**n))) @ (np.kron(np.identity(2**n), deltaMatrix(a)) @ np.kron(deltaMatrix(a), np.identity(2**n)))
    #transf_rho = transf @ rho

    #print(rho[a::2**n])
    #print(rho[a*2**n:(a+1)*2**n])
    #print(rho[b*2**n + b])

    #rho_dot[a*2**n:(a+1)*2**n] += pre_factor * rho[a*2**n:(a+1)*2**n]   # Column a of rho matrix
    #rho_dot[b*2**n + b]


    #print("transformed rho:\n", transf_rho)
    return None



T = 0.008
lam_sq = 0.00001
n = 3 # number of qbits




initial_rho = (1/n**0.5) * np.ones((2**n, 2**n))
initial_phi = (1/(2**n)**0.5) * np.ones(2**n)
#initial_rho = np.zeros((2**n,2**n))
#initial_rho[3,3] = 1

vec_initial_rho = initial_rho.flatten(order='F')

random_state1 = np.ones((2**n, 2**n))
random_state2 = np.identity(2**n)
random_state3 = np.arange(4**n).reshape((2**n,2**n))


minplusPauli(2, random_state3, act_from_left=False, minus=False)
#xPauli(n, 1, random_state2, act_from_left=True)
#LindBladian(operator=True, rho=vec_initial_rho)
#CrankNicholsan(vec_initial_rho)
#DirectMethod(vec_initial_rho)
#MCWF(initial_phi)
#RemoveGroundState(2, initial_phi)
#TestFcn(vec_initial_rho)

#print(rho_initial)
#print(EigenStates(n))

