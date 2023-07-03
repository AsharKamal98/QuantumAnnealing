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

def Hamiltoneon():
    """ Constructs Hamiltoneon specificially for n=2 qbits case """
    sz = np.array([[1,0],[0,-1]])
    I = np.identity(n)
    H = np.kron(sz, I) + np.kron(I, sz)
    return H

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
    N = 1/(math.exp(beta*(x-y))-1) if x > y else 0
    return N


def g_old(eigenvalues, b, a):
    g = lam_sq if eigenvalues[b] > eigenvalues[a] else 0
    return g

def g(x, y):
    g = lam_sq if x > y else 0
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


def MCWF(phi):
    print("initial phi\n", phi)

    dt = 0.00001 
    t = 1
    iterations = int(t/dt)

    eigenstates, eigenvalues = EigenSystem()

    conv = np.zeros((2**n,iterations))
    convv = np.zeros((6,iterations))

    #emission_pre_factors, absorption_pre_factors = PreFactors2()
    #print("emission prefactors:", emission_pre_factors)
    #print("absorption prefactors:", absorption_pre_factors)

    for i in range(iterations):
        emission_pre_factors = []
        absorption_pre_factors = []

        delta_p_list = []
        for j in range(n):
            # Spontaneous photon emission by qbit j+1 (qbit counting starts at 1, forloop starts at 0)
            phi_b = minplusPauli(j+1, phi, minus=True)
            phi_a = minplusPauli(j+1, phi_b, minus=False)
            energy_b, energy_a = ((np.conj(phi_b)*phi_b) @ eigenvalues).real, ((np.conj(phi_a)*phi_a) @ eigenvalues).real
            emission_pre_factors.append((N(energy_a, energy_b)+1) * g(energy_a, energy_b))
            #delta_p_list.append(emission_pre_factors[j] * (np.conj(phi) @ minplusPauli(j+1, minplusPauli(j+1, phi, minus=True), minus=False)) * dt)
            delta_p_list.append(emission_pre_factors[j] * (np.conj(phi_a) @ phi_a) * dt)

            # Absorption of photon by qbit j+1
            phi_b = minplusPauli(j+1, phi, minus=False)
            phi_a = minplusPauli(j+1, phi_b, minus=True)
            energy_b, energy_a = ((np.conj(phi_b)*phi_b) @ eigenvalues).real, ((np.conj(phi_a)*phi_a) @ eigenvalues).real
            absorption_pre_factors.append(N(energy_b, energy_a) * g(energy_b, energy_a))
            #delta_p_list.append(absorption_pre_factors[j] * (np.conj(phi) @ minplusPauli(j+1, minplusPauli(j+1, phi, minus=False), minus=True)) * dt)
            delta_p_list.append(absorption_pre_factors[j] * (np.conj(phi_a) @ phi_a) * dt)


        delta_p = np.sum(delta_p_list)
        epsilon = random.rand()

        if delta_p > 0.1:
            print("Warning! delta_p is getting large, must be much smaller than 1. Current value:", delta_p)

        if epsilon > delta_p:   # -> No emission/absorption
            phi_1 = phi - complex(0,1) * (zPauli(1,phi) + zPauli(2,phi)) * dt 
            for j in range(n):
                phi_1 -= (1/2) * emission_pre_factors[j] * minplusPauli(j+1, minplusPauli(j+1, phi, minus=True), minus=False) * dt
                phi_1 -= (1/2) * absorption_pre_factors[j] * minplusPauli(j+1, minplusPauli(j+1, phi, minus=False), minus=True) * dt
            phi_new = (phi_1.copy())/((1-delta_p)**0.5)
            #phi_new = phi_1/np.linalg.norm(phi_1)

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

            #if np.linalg.norm(phi_new) < 0.1:
            #    print("norm zero")
            #print("max_prob", max_prob)
            #print("pre_factors:", emission_pre_factors[site], absorption_pre_factors[site])
            #print("phi", phi_new)

        conv[:,i] = np.absolute(phi_new-phi)
        convv[0,i] = (phi_new[-1]).real
        convv[1,i] = (phi_new[-1]).imag
        convv[2,i] = np.absolute(phi_new[-1])
        convv[3,i] = (phi_new[-2]).real
        convv[4,i] = (phi_new[-2]).imag
        convv[5,i] = np.absolute(phi_new[-2])

        phi = phi_new

        phi_len = np.linalg.norm(phi)
        #if phi_len > 1.5 or phi_len < 0.5:
            #print("phi not normalized properly. Its norm is\n", phi_len)
    

    for index in range (2**n):
        plt.plot(np.linspace(0,iterations-1,iterations), conv[index], 'ro', markersize=2)
        plt.title("phi_new - phi")
        plt.show()


    plt.plot(np.linspace(0,iterations-1,iterations), convv[0], 'go', markersize=1, label="real")
    plt.plot(np.linspace(0,iterations-1,iterations), convv[1], 'bo', markersize=1, label="imag")
    plt.plot(np.linspace(0,iterations-1,iterations), convv[2], 'ro', markersize=1, label="norm")
    plt.legend()
    plt.show()
    plt.plot(np.linspace(0,iterations-1,iterations), convv[3], 'go', markersize=1, label="real")
    plt.plot(np.linspace(0,iterations-1,iterations), convv[4], 'bo', markersize=1, label="imag")
    plt.plot(np.linspace(0,iterations-1,iterations), convv[5], 'ro', markersize=1, label="norm")
    plt.legend()
    plt.show()


    print("new phi\n", phi.round(decimals=3))
    print("phi length:", phi_len)

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


def PreFactors(phi):
    """ Computes pre_factors of all operators (emission and absorption operators for all sites). """

    eigenstates,eigenvalues = EigenSystem()

    emission_pre_factors = []
    absorption_pre_factors = []



    return emission_pre_factors, absorption_pre_factors


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



T = 1000#0.008#0.008
lam_sq = 1
n = 2 # number of qbits




initial_rho = (1/n**0.5) * np.ones((2**n, 2**n))
initial_phi = (1/(2**n)**0.5) * np.ones(2**n)
#initial_rho = np.zeros((2**n,2**n))
#initial_rho[3,3] = 1

vec_initial_rho = initial_rho.flatten(order='F')

random_state1 = np.ones((2**n, 2**n))
random_state2 = np.identity(2**n)
random_state3 = np.arange(4**n).reshape((2**n,2**n))

#print("initial", initial_phi)
#print("first", zPauli(1,initial_phi))
#print("second", zPauli(2,initial_phi))
#print("sum", zPauli(1,initial_phi) + np.array(zPauli(2,initial_phi)))
#print(initial_phi + initial_phi)

#minplusPauli(2, minplusPauli(2, initial_phi, minus=False), minus=True)
#minplusPauli(2, random_state3, act_from_left=False, minus=False)
#xPauli(n, 1, random_state2, act_from_left=True)
#LindBladian(operator=True, rho=vec_initial_rho)
#CrankNicholsan(vec_initial_rho)
#DirectMethod(vec_initial_rho)
MCWF(initial_phi)
#RemoveGroundState(2, initial_phi)
#TestFcn(vec_initial_rho)

#print(rho_initial)
#print(EigenStates(n))

