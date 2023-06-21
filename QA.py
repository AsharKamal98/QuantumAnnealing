import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
import time


def zPauli(site, state, act_from_left):
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

#n=2
#state = np.ones((4,4))
#zPauli(2,state,True)


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

#n=2
#state = np.zeros((4,4))
#state[1,1]=1
#state[2,3]=1
#state[1,0]=1
#xPauli(2,2,state,False)


def minplusPauli(site, state, act_from_left, minus):
    """
    Applies Reasing/lowering operator sigma_(plus/minus) on a given state.

    INPUT:
    ------
    site =          site =  integer(1,2,...). The spin/qbit at which Pauli matrix applied 
    state =         array(s) of size 2**n. If multiple arrays are provided, place them into a matrix.
    act_from_left = boolean. If True, Pali matrix acts on on ket.
    minus =         boolean. If True, lowering operator applied, else raising operator. 
    """

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
    
#n=2
#state = np.array([0,0,1,0])
#minplusPauli(1,state,act_from_left=True,minus=False)



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
    """ Creates a matrix of zeros, except for a 1 in the diagonal element [index,index] """
    delta = np.zeros((2**n,2**n))
    delta[index,index] = 1
    return delta

def PermMatrix(old_index, new_index):
    E = np.identity(2**n)
    temp = E[old_index].copy()
    E[old_index] = E[new_index]
    E[new_index] = temp
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
    else:
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
                rho_dot += np.kron(np.identity(2**n), deltaMatrix(a))
                rho_dot += np.kron(deltaMatrix(a), np.identity(2**n))

                E = PermMatrix(a,b)
                delta = deltaMatrix(a)
                rho_dot -= (np.kron(np.identity(2**n), E) @ np.kron(E, np.identity(2**n))) @ (np.kron(np.identity(2**n), deltaMatrix(a)) @ np.kron(deltaMatrix(a), np.identity(2**n)))

            else:
                rho_dot[a::2**n] += pre_factor * rho[a::2**n]                       # Row a of rho matrix
                rho_dot[a*2**n:(a+1)*2**n] += pre_factor * rho[a*2**n:(a+1)*2**n]   # Column a of rho matrix
                rho_dot[b*2**n + b] -= pre_factor * rho[a*2**n + a]                 # a/b'th diagonal element of rho matrix

    return rho_dot


def CrankNicholsan(rho):
    dt=0.1
    iterations = 20
    conv = np.zeros((4**n,iterations))

    print("initial rho:\n", rho)
    for i in range(iterations):
        # RHS of Crank Nicholsan
        rho_dot = LindBladian(operator=False, rho=rho)
        print(np.absolute(rho_dot))
        rho_prime_old = rho - (1/2)*complex(0,1)*rho_dot*dt
        
        # LHS of Crank Nicholsan
        rho_dot_operator = LindBladian(operator=True)
        rho_new = np.linalg.solve((1 + (1/2)*complex(0,1)*rho_dot_operator*dt), rho_prime_old)

        conv[:,i] = np.absolute(rho_new-rho) #np.absolute(rho_dot)
        rho = rho_new.copy()

        #if i%(int(iterations/5))==0:
        #print("\nstate vector: \n", rho.round(decimals=3))


    for index in range (4**n):
        if index < 1:
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


def TestFcn(rho):
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
lam_sq = 1
n = 2 # number of qbits




initial_rho = (1/n**0.5) * np.ones((2**n, 2**n))
#initial_rho = np.arange(0,4**n).reshape((2**n,2**n))
#print("initial rho:\n", initial_rho)
#initial_rho = np.zeros((2**n,2**n))
#initial_rho[3,3] = 1

vec_initial_rho = initial_rho.flatten(order='F')

random_state1 = np.ones((2**n, 2**n))
random_state2 = np.identity(2**n)
random_state3 = np.ones(2**n)

#minplusPauli(n, 1, random_state3, act_from_left=True, minus=True)
#xPauli(n, 1, random_state2, act_from_left=True)
#LindBladian(operator=True, rho=vec_initial_rho)
#CrankNicholsan(vec_initial_rho)
TestFcn(vec_initial_rho)

#print(rho_initial)
#print(EigenStates(n))

