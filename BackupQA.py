import numpy as np
from numpy import random
import cmath
import math
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time
from sympy import *
import sys
from tqdm import tqdm
import random

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
    transformed_state = state.copy()
    for index in range(2**n):
        # Finds bit value corresponding to the site
        k = n-site
        bit_value = (index & (1 << k)) >> k

        # If bit value is 1, a sign change should occur
        if bit_value==1:
            if act_from_left:
                transformed_state[index] = -state[index]
            else:
                transformed_state[:,index] = -state[:,index]

        #print("state after:\n", state)

    return transformed_state


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
    state =         arrays of size (2**n,2**n) (density matrix) / (2**n) (state vector).
                    If you act on a state vector (2**n array), do not specify act_from_left variable.
    act_from_left = boolean. If True, Pali matrix acts on on kets of density matrix (state variable).
    minus =         boolean. If True, lowering operator applied, else raising operator. 
    """

    #print("state before: \n", state)
    transformed_state = np.zeros(state.shape, dtype=complex)
    for index in range(2**n):
        # Finds bit value corresponding to the site
        k = n-site
        bit_value = (index & (1 << k)) >> k

        # Toggle bit value
        if act_from_left:
            if not minus and bit_value==1:
                new_index = index-2**(n-site) # bit 1 -> 0
                transformed_state[new_index] += state[index]
            elif minus and bit_value==0:
                new_index = index+2**(n-site) # bit 0 -> 1
                transformed_state[new_index] += state[index]

        else:
            if not minus and bit_value==0:
                new_index = index+2**(n-site)
                transformed_state[:,new_index] += state[:,index]
            elif minus and bit_value==1:
                new_index = index-2**(n-site)
                transformed_state[:,new_index] += state[:,index]


    #print("state after: \n", transformed_state)

    return transformed_state

def C(a,b, phii):
    phi_prime = np.zeros(phii.shape[0], dtype=complex)
    phi_prime[b] = phii[a].copy()
    return phi_prime
         

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
    #eigenvalues = [0,2,-2,0]
    #eigenvalues = [0,-2,2,0]
    return eigenstates, eigenvalues

def HamiltoneonOperator():
    """ Constructs Hamiltoneon specificially for n=2 qbits case """
    sz = np.array([[1,0],[0,-1]])
    I = np.identity(n)
    H = np.kron(sz, I) + np.kron(I, sz)
    return H

def Hamiltoneon(s, n):
    sz = np.array([[1,0],[0,-1]])
    sx = np.array([[0,1],[1,0]])
    I = np.identity(2)

    if n==2:
        H = (np.kron(sx, I) + np.kron(I, sx))*(1-s) + (np.kron(sz, I) + np.kron(I, sz))*s
    elif n==3:
        H = (np.kron(sx, np.kron(I, I)) + np.kron(I, np.kron(sx, I)) + np.kron(I, np.kron(I, sx)))*(1-s) + \
            (np.kron(sz, np.kron(I, I)) + np.kron(I, np.kron(sz, I)) + np.kron(I, np.kron(I, sz)))*s
    elif n==4:
        H = (np.kron(sx, np.kron(I, np.kron(I, I))) + np.kron(I, np.kron(sx, np.kron(I, I))) + np.kron(I, np.kron(I, np.kron(sx, I))) + np.kron(I, np.kron(I, np.kron(I, sx))))*(1-s) + \
            (np.kron(sz, np.kron(I, np.kron(I, I))) + np.kron(I, np.kron(sz, np.kron(I, I))) + np.kron(I, np.kron(I, np.kron(sz, I))) + np.kron(I, np.kron(I, np.kron(I, sz))))*s


    eigenvalues, eigenstates = np.linalg.eigh(H)
    #eigenstates = np.array([eigenstate/np.linalg.norm(eigenstate) for eigenstate in eigenstates]) # Needed?
    
    #print("Hamiltoneon:\n", H)
    #print("Eigenvectors:\n", eigenstates.T)
    #print("Eigenvalues:\n", eigenvalues)

    return H, eigenstates.T, eigenvalues



def Hcommutator(rho):
    """ Computes [H, \rho] for H = \sigma_z^1 + \sigma_z^2 """
    commutator = zPauli(n, 1, rho, act_from_left=True) + zPauli(n, 2, rho, act_from_left=True) - \
            zPauli(n, 1, rho, act_from_left=False) - zPauli(n, 2, rho, act_from_left=False)
    return commutator

def N_old(eigenvalues, b, a):
    beta = 1/T
    N = 1/(math.exp(beta*(eigenvalues[b]-eigenvalues[a]))-1) if eigenvalues[b] > eigenvalues[a] else 0
    #N = 1 if eigenvalues[b] > eigenvalues[a] else 0
    return N

def N(x, y):
    beta = 1/T
    if x < y:
        sys.exit("Eigenvalues incorrect order in N function")
    if x.imag > 0.01 or y.imag > 0.01:
        sys.exit("complex eigenvalues in N function!")

    N = 1/(math.exp(beta*(abs(x-y)))-1) if round(abs(x-y), 4)>0 else 0
    return N


def g_old(eigenvalues, b, a):
    g = lam_sq if eigenvalues[b] > eigenvalues[a] else 0
    return g

def g(x, y):
    #g = lam_sq if x > y else 0
    g = lam_sq
    return g


def deltaMatrix(index):
    """ Creates a matrix of zeros, except for a 1 in the diagonal element [index,index].
        This is used to pick out certain elements of a matrix (rho). """
    delta = np.zeros((2**n,2**n))
    delta[index,index] = 1
    return delta

def PermMatrix(a, b):
    """ Creates an identity matrix where the a and b rows are swapped. 
        This is used to move elements of a matrix (rho) along the diagonal. """
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
                pre_factor -= Nba * abs(gba)**2 * np.matmul(a_state, minplusPauli(i+1, b_state, minus=True)) * \
                        np.matmul(b_state, minplusPauli(i+1, a_state, minus=False)) + \
                        (Nab+1) * abs(gab)**2 * np.matmul(b_state, minplusPauli(i+1, a_state, minus=True)) * \
                        np.matmul(a_state, minplusPauli(i+1, b_state, minus=False))

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


def MCWF2(phi):
    print("initial phi\n", phi)
    dt_ds_ratio = 0.00015
    ds = 0.000005 #0.000005
    dt = dt_ds_ratio * ds #0.0001
    #iterations = int(1.15*int(1/ds))
    iterations = int(1/ds)
    iterations2 = 3

    #t = 10
    #dt = 0.0001
    #iterations = int(t/dt)

    conv = np.zeros((2**n,iterations))
    convv = np.ones((3*2**n,iterations))*99

    for i in tqdm(range(iterations)):
        s = min(i * ds, 1)
        #s = 1

        H, H_diag, eigenstates, eigenvalues = Hamiltoneon(s)
        for k in range(iterations2):
            emission_pre_factors = []
            absorption_pre_factors = []
            pre_factors = []
            delta_p_list = []
            for j in range(n):
                # Spontaneous photon emission by qbit j+1 (qbit counting starts at 1, forloop starts at 0)
                phi_b = minplusPauli(j+1, phi, minus=True)
                phi_a = minplusPauli(j+1, phi_b, minus=False)
                #phi_b_decomp = np.linalg.lstsq(eigenstates.T, phi_b, rcond=None)[0]
                #print("------------------------------")
                #print("phi:", np.linalg.norm(phi), "\n", phi.round(3))
                #print("phi_b", np.linalg.norm(phi_b), "\n", phi_b.round(3))
                #print("eigenstates\n", eigenstates.round(3))
                #print("phi_b_decomp", np.linalg.norm(phi_b_decomp), "\n", phi_b_decomp.round(3))
                #print("-----------------------------")
                #if np.linalg.norm(phi_b_decomp) > 0:
                #phi_b_decomp *= np.linalg.norm(phi_b)/np.linalg.norm(phi_b_decomp)
                #else:
                #    phi_b_decomp *= 0
                #phi_a_decomp = np.linalg.lstsq(eigenstates.T, phi_a, rcond=None)[0]
                #if np.linalg.norm(phi_a_decomp) > 0:
                #phi_a_decomp *= np.linalg.norm(phi_a)/np.linalg.norm(phi_a_decomp)
                #else:
                #    phi_a_decomp *= 0
                energy_b, energy_a = ((np.conj(phi_b)*phi_b) @ eigenvalues).real, ((np.conj(phi_a)*phi_a) @ eigenvalues).real
                #if abs(energy_a) > 2.3 or abs(energy_b) > 2.3:
                #    print("phi:", np.linalg.norm(phi), "\n", phi.round(3))
                #    print("phi_b", np.linalg.norm(phi_b), "\n", phi_b.round(3))
                #    print("phi_a", np.linalg.norm(phi_a), "\n", phi_a.round(3))
                #    print("eigenvalues\n", eigenvalues)
                #    print("energies\n", energy_b, energy_a)
                #    sys.exit("energy diverging!")
                if energy_a > energy_b: 
                    emission_pre_factors.append((N(energy_a, energy_b)+1) * g(energy_a, energy_b))
                    #pre_factors.append((N(energy_a, energy_b)+1) * g(energy_a, energy_b))
                else: 
                    emission_pre_factors.append(N(energy_a, energy_b) * g(energy_a, energy_b))
                    #pre_factors.append(N(energy_a, energy_b) * g(energy_a, energy_b))

                #delta_p_list.append(emission_pre_factors[j] * (np.conj(phi) @ minplusPauli(j+1, minplusPauli(j+1, phi, minus=True), minus=False)) * dt)
                delta_p_list.append(emission_pre_factors[j] * (np.conj(phi_a) @ phi_a) * dt)

                # Absorption of photon by qbit j+1
                phi_b = minplusPauli(j+1, phi, minus=False)
                phi_a = minplusPauli(j+1, phi_b, minus=True)

                #phi_b_decomp = np.linalg.lstsq(eigenstates.T, phi_b, rcond=None)[0]
                #if np.linalg.norm(phi_b_decomp) > 0:
                #phi_b_decomp *= np.linalg.norm(phi_b)/np.linalg.norm(phi_b_decomp)
                #else:
                #    phi_b_decomp *= 0
                #phi_a_decomp = np.linalg.lstsq(eigenstates.T, phi_a, rcond=None)[0]
                #if np.linalg.norm(phi_b_decomp) > 0:
                #phi_a_decomp *= np.linalg.norm(phi_a)/np.linalg.norm(phi_a_decomp)
                #else:
                #    phi_a_decomp *= 0
                energy_b, energy_a = ((np.conj(phi_b)*phi_b) @ eigenvalues).real, ((np.conj(phi_a)*phi_a) @ eigenvalues).real
                if energy_a > energy_b:
                    absorption_pre_factors.append((N(energy_a, energy_b)+1) * g(energy_a, energy_b))
                else:
                    absorption_pre_factors.append(N(energy_a, energy_b) * g(energy_a, energy_b))
                #delta_p_list.append(absorption_pre_factors[j] * (np.conj(phi) @ minplusPauli(j+1, minplusPauli(j+1, phi, minus=False), minus=True)) * dt)
                delta_p_list.append(absorption_pre_factors[j] * (np.conj(phi_a) @ phi_a) * dt)


            delta_p = np.sum(delta_p_list)
            epsilon = random.rand()
            epsilon = 10
    
            if delta_p > 0.1:
                print("Warning! delta_p is getting large, must be much smaller than 1. Current value:", delta_p)

            if epsilon > delta_p:   # -> No emission/absorption
                #phi_1 = phi - complex(0,1) * (zPauli(1,phi) + zPauli(2,phi)) * dt 
                phi_1 = phi - complex(0,1) * (H_diag @ phi) * dt
                for m in range(n):
                    phi_1 -= (1/2) * emission_pre_factors[m] * minplusPauli(m+1, minplusPauli(m+1, phi, minus=True), minus=False) * dt
                    phi_1 -= (1/2) * absorption_pre_factors[m] * minplusPauli(m+1, minplusPauli(m+1, phi, minus=False), minus=True) * dt
                phi_new = (phi_1.copy())/((1-delta_p)**0.5)

            else:
                max_prob = max(delta_p_list)
                index = np.argmax(delta_p_list)
                site = int(index/2) # Counting starts at zero here
                if index%2==0:
                    print("spontaneous emission!")
                    phi_new = emission_pre_factors[site]**0.5 * minplusPauli(site+1, phi, minus=True)/((max_prob/dt)**0.5)
                else:
                    print("absorption!")
                    phi_new = absorption_pre_factors[site]**0.5 * minplusPauli(site+1, phi, minus=False)/((max_prob/dt)**0.5)
       

            #phi_norm = np.linalg.norm(phi_new)
            #if phi_norm < 0.9 or phi_norm > 1.1:
                #print("norm diverging from 1:", phi_norm)


            #conv[:,i] = np.absolute(phi_new-phi)
            for l in range(2**n):
                index = l*3
                convv[index,i] = (phi_new[l]).real
                convv[index+1,i] = (phi_new[l]).imag
                convv[index+2,i] = np.absolute(phi_new[l])


            phi = phi_new

        phi_len = np.linalg.norm(phi)
        if phi_len > 1.3 or phi_len < 0.7:
            print("phi norm:", phi_len)
            sys.exit("phi not normalized properly")



    fig, ax = plt.subplots(2,2, sharex=True, sharey=False, figsize=(15,11))
    iteration_list = np.linspace(0,iterations-1,iterations)
    nth_element = 10
    for j in range(2**n):
        index = j*3
        grid_index1 = 0 if (j==0 or j==1) else 1
        grid_index2 = 0 if (j==0 or j==2) else 1
        

        ax[grid_index1, grid_index2].plot(iteration_list[::nth_element], convv[index][::nth_element], 'go', markersize=0.3, label="real")
        ax[grid_index1, grid_index2].plot(iteration_list[::nth_element], convv[index+1][::nth_element], 'bo', markersize=0.3, label="imag")
        ax[grid_index1, grid_index2].plot(iteration_list[::nth_element], convv[index+2][::nth_element], 'ro', markersize=0.3, label="norm")
        ax[grid_index1, grid_index2].axvline(x=1/ds, color='purple')
        ax[grid_index1, grid_index2].set_ylim(-1.05, 1.05)
        ax[grid_index1, grid_index2].set_title("Phi component {}".format(j+1))
    ax[1, 1].legend()
    plt.savefig('AdiabaticAnnealing2.png', bbox_inches='tight')


    print("absolute phi\n", np.absolute(phi).round(decimals=3))
    print("phi norm:", np.linalg.norm(phi))

    return None

def PreFactors2():
    """ Computes pre_factors of all operators (emission and absorption operators for all sites). """

    eigenstates,eigenvalues = EigenSystem()

    emission_pre_factors = []
    absorption_pre_factors = []

    for i in range(n):
        emission_pre_factor = 0
        absorption_pre_factor = 0
        for a in range(2**n):
            a_state = eigenstates[a]
            for b in range(2**n):
                b_state = eigenstates[b]

                Nba = N_old(eigenvalues, b, a)
                gba = g_old(eigenvalues, b, a)
                Nab = N_old(eigenvalues, a, b)
                gab = g_old(eigenvalues, a, b)
        

                emission_pre_factor += (Nab+1) * abs(gab)**2 * np.matmul(b_state, minplusPauli(i+1, a_state, minus=True)) * \
                            np.matmul(a_state, minplusPauli(i+1, b_state, minus=False))
    
                absorption_pre_factor += Nba * abs(gba)**2 * np.matmul(a_state, minplusPauli(i+1, b_state, minus=True)) * \
                            np.matmul(b_state, minplusPauli(i+1, a_state, minus=False))

        emission_pre_factors.append(emission_pre_factor)
        absorption_pre_factors.append(absorption_pre_factor)

    return emission_pre_factors, absorption_pre_factors

def MCWF(phi, n, plot_phi_history=True):
    #print("initial phi\n", phi)
    dt_ds_ratio = 2
    ds = 0.000025 
    dt = dt_ds_ratio * ds 
    #iterations_s = int(1.15*int(1/ds))
    iterations_s = int(1/ds)
    iterations_t = 10

    phi_history_z = np.ones((3*2**n,iterations_s))*99
    phi_history_x = np.ones((3*2**n,iterations_s))*99

    for i in tqdm(range(iterations_s)):
        s = min(i * ds, 1)
        H, eigenstates, eigenvalues = Hamiltoneon(s, n)
        phi_decomp = np.array([np.conj(phi) @ eigenstate for eigenstate in eigenstates])
        phi_decomp_norm = np.linalg.norm(phi_decomp)
        if phi_decomp_norm > 1.3:
            print(phi_decomp_norm)
            sys.exit("phi_decomp norm diverging")
        for j in range(iterations_t):
            # COMPUTE PRE-FACTORS AND \delta_p
            pre_factors = []
            delta_p_list = []
            counter_list = []
            energy_list = []
            for a in range(2**n):
                energy_a = eigenvalues[a]
                for b in range(2**n):
                    energy_b = eigenvalues[b]
                    if a==b:
                        continue
                    # Spontaneous emission
                    if energy_a > energy_b:
                        pre_factors.append((N(energy_a, energy_b)+1) * g(energy_a, energy_b))
                        photon_type = "em"
                    # Absorption
                    elif energy_a < energy_b:
                        pre_factors.append(N(energy_b, energy_a) * g(energy_a, energy_b))
                        photon_type = "abs"
                    # Degenerate eigenvalues (energy_a = energy_b)
                    else:
                        pre_factors.append(0)
                        photon_type = "None"


                    delta_p_list.append(pre_factors[-1] * (np.conj(phi_decomp) @ C(b,a, C(a,b,phi_decomp))) * dt) # Can be made faster!
                    counter_list.append([a,b, photon_type])
                    energy_list.append([energy_a, energy_b])

            delta_p = np.sum(delta_p_list)
            epsilon = random.random()

            if delta_p > 0.1:
                print("Warning! delta_p is getting large, must be much smaller than 1. Current value:", delta_p)

            if epsilon > delta_p:   # -> No emission/absorption
                phi_1 = phi_decomp - complex(0,1) * (H @ phi_decomp) * dt
                counter = 0
                for a in range(2**n):
                    energy_a = eigenvalues[a]
                    for b in range(2**n):
                        energy_b = eigenvalues[b]
                        if a==b:
                            continue
                        phi_1 -= (1/2) * pre_factors[counter] * C(b,a, C(a,b,phi_decomp)) * dt # Can be made faster!
                        counter += 1
                phi_new = (phi_1.copy())/((1-delta_p)**0.5)

            else:
                delta_p_list = np.real(delta_p_list)
                index = random.choices(range(len(delta_p_list)), delta_p_list)[0]
                delta_p_m = delta_p_list[index]
                a, b, photon_type = counter_list[index]
                #if photon_type == "em":
                #    print("spontaneous emission!")
                #elif photon_type == "abs":
                #    print("absorption!")
                #else:
                #    print("Very odd")
                phi_new = pre_factors[index]**0.5 * (C(a,b,phi_decomp.copy())/((delta_p_m/dt)**0.5))
                #print("phi before", phi_decomp.round(decimals=2))
                #print("phi after", phi_new.round(decimals=2))
                #print("energy_a", energy_list[index][0])
                #print("energy_b", energy_list[index][1])
                #print("energy difference", energy_list[index][0]-energy_list[index][1])
            phi_decomp = phi_new.copy()

        # Back to old basis
        phi = eigenstates.T @ phi_decomp

        if plot_phi_history:
            for k in range(2**n):
                index = k*3
                phi_history_z[index,i] = (phi[k]).real
                phi_history_z[index+1,i] = (phi[k]).imag
                phi_history_z[index+2,i] = np.absolute(phi[k])

                phi_history_x[index,i] = (phi_decomp[k]).real
                phi_history_x[index+1,i] = (phi_decomp[k]).imag
                phi_history_x[index+2,i] = np.absolute(phi_decomp[k])



        phi_len = np.linalg.norm(phi)
        if phi_len > 1.3 or phi_len < 0.7:
            print("phi norm:", phi_len)
            #sys.exit("phi not normalized properly")
            print("phi not normalized properly #################################################")
            return np.zeros(2**n)


    if plot_phi_history:
        #fig_z, ax_z = plt.subplots(2,4, sharex=True, sharey=False, figsize=(20,10))
        #fig_x, ax_x = plt.subplots(2,4, sharex=True, sharey=False, figsize=(20,10))
        fig_z, ax_z = plt.subplots(4,4, sharex=True, sharey=False, figsize=(10,10))
        fig_x, ax_x = plt.subplots(4,4, sharex=True, sharey=False, figsize=(10,10))
        iteration_list = np.linspace(0,iterations_s-1,iterations_s)
        nth_element = 2
        for j in range(2**n):
            index = j*3
            grid_index1 = int(j/4) #0 if j<4 else 1
            grid_index2 = j%4

            ax_z[grid_index1, grid_index2].plot(iteration_list, phi_history_z[index], 'go', markersize=1, label="real")
            ax_z[grid_index1, grid_index2].plot(iteration_list, phi_history_z[index+1], 'bo', markersize=1, label="imag")
            ax_z[grid_index1, grid_index2].plot(iteration_list, phi_history_z[index+2], 'ro', markersize=1, label="norm")
            ax_z[grid_index1, grid_index2].axvline(x=1/ds, color='purple')
            ax_z[grid_index1, grid_index2].set_ylim(-1.05, 1.05)
            ax_z[grid_index1, grid_index2].set_title("Phi component {}".format(j+1))

            ax_x[grid_index1, grid_index2].plot(iteration_list, phi_history_x[index], 'go', markersize=1, label="real")
            ax_x[grid_index1, grid_index2].plot(iteration_list, phi_history_x[index+1], 'bo', markersize=1, label="imag")
            ax_x[grid_index1, grid_index2].plot(iteration_list, phi_history_x[index+2], 'ro', markersize=1, label="norm")
            ax_x[grid_index1, grid_index2].axvline(x=1/ds, color='purple')
            ax_x[grid_index1, grid_index2].set_ylim(-1.05, 1.05)
            ax_x[grid_index1, grid_index2].set_title("Phi component {}".format(j+1))

        ax_z[1,1].legend()
        ax_x[1,1].legend()
        fig_z.savefig('AdiabaticAnnealing_z.png', bbox_inches='tight')
        fig_x.savefig('AdiabaticAnnealing_x.png', bbox_inches='tight')


    print("absolute phi_decomp\n", np.absolute(phi_decomp).round(decimals=3))
    #print("absolute phi\n", np.absolute(phi).round(decimals=3))
    print("phi norm:", np.linalg.norm(phi))

    return np.absolute(phi_decomp)


def BoltzmanCheck(initial_phi, n, run_MCWF=False):
    filename = "AverageProb2Q.txt"
    if run_MCWF:
        statistics = []
        iterations = 30
        for i in tqdm(range(iterations)):
            phi = MCWF(initial_phi, n, plot_phi_history=False)
            statistics.append(phi)
        probabilities = np.absolute(np.array(statistics)).round(decimals=10)

        with open(filename, "a") as f:
            for i in range(iterations):
                for j in range(2**n):
                    f.writelines(f'{probabilities[i,j]:<{15}}')
                f.writelines("\n")

    with open(filename, "r") as f:
        l = f.readlines()
        probabilities = np.array([l[i].split() for i in range(len(l))])
    probabilities = probabilities.astype(np.float64)

    std_dev = np.std(probabilities,axis=0).round(decimals=10)
    std_error = std_dev/(np.sqrt(len(l)))
    avg_prob = np.sum(probabilities, axis=0)/len(l)

    print("------------------------- SUMMARY ----------------------------")
    print("All probabilities\n", probabilities.round(decimals=3))
    print("Standard deviation\n", std_dev.round(decimals=3))
    print("Standard error\n", std_error.round(decimals=3))
    print("Averaged probabilities\n", avg_prob.round(decimals=3))
    print("--------------------------  END ------------------------------")
    return None



T = 1#0.008#0.008
lam_sq = 1
num_qbits = 2


#initial_phi = np.array([0.5, -0.5, -0.5, 0.5], dtype=complex)
H, eigenstates, eigenvalues = Hamiltoneon(0, num_qbits)
initial_phi = eigenstates[0]

#vec_initial_rho = initial_rho.flatten(order='F')

#random_state1 = np.ones((2**n, 2**n))
#random_state2 = np.identity(2**n)
#random_state3 = np.arange(4**n).reshape((2**n,2**n))

#print("initial", initial_phi)
#print("first", zPauli(1,initial_phi))
#print("second", zPauli(2,initial_phi))
#print("sum", zPauli(1,initial_phi) + np.array(zPauli(2,initial_phi)))
#print(initial_phi + initial_phi)

#minplusPauli(2, minplusPauli(2, initial_phi, minus=False), minus=True)
#minplusPauli(2, random_state3, act_from_left=False, minus=False)
#xPauli(n, 1, random_state2, act_from_left=True)
#Hamiltoneon(0)
#LindBladian(operator=True, rho=vec_initial_rho)
#CrankNicholsan(vec_initial_rho)
#DirectMethod(vec_initial_rho)
#MCWF(initial_phi, num_qbits)
BoltzmanCheck(initial_phi, num_qbits, run_MCWF=True)
#RemoveGroundState(2, initial_phi)
#TestFcn(vec_initial_rho)

#print(rho_initial)
#print(EigenStates(n))
