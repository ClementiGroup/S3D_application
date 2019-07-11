# This script is to perform space time diffusion map calculation for each metastable state

#--------------------------------------------------------------------

# INPUT needed:
        # discrete trajectory obtained with k-means clustering
        # number of hidden state and the lag time for HMM estimation (validated in 2F4K_analysis.ipynb)
        # simulation trajectory with topology file

# OUTPUT: the eigenvalues and eigenvectors of the diffusion map

#--------------------------------------------------------------------

# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pyemma
from pyemma import msm, plots    # pyemma APIs
import scipy
import scipy.sparse as sps
import time
import scipy.sparse.linalg as spl
import sklearn.neighbors as neigh_search
import sklearn.cluster as skl_cl
import sys

#--------------------------------------------------------------------
# set some path variables
path_results = './Diffusion_Map/'    # where clustering results should be saved
path_discrete_traj = './MSM/'   # where the discrete trajectory .npz file sits
path_full_traj = './traj/'       # where the full trajectory sits

dtraj = np.load(path_discrete_traj + '2F4K_MSM_10TICA_clusters_1000.npz')
dtraj.keys()

n_clusters = dtraj['n_clusters']
discrete_traj = dtraj['micro_membership']

n_traj = 1

for k in range(n_traj):
    print('configurations in trajectory ' + str(k) + ' equal to ' + str(np.shape(discrete_traj[k])))
sys.stdout.flush()

dtrajs = [discrete_traj[0]]

#--------------------------------------------------------------------
# preparing loading trajectories with the featurizer
import mdtraj as md
# parameters
nskip = 1
print('frames to be skipped between successive configurations = ' + str(nskip))
delta = 100.0/(500000.0/nskip)
print('timestep between successive configurations             = ' + str(delta))
n_traj = 1
print('number of traj chunks per trajectory (DE Shaw)         = ' + str(n_traj))
sys.stdout.flush()

# collect all file names in a list...then define featurizer to actually read in trajectories

allfiles1 = [path_full_traj + '2F4K-0-protein_all.dcd']
topology = path_full_traj + '2F4K-0-protein_fixed_noH.pdb'

import pyemma.coordinates as coor
import itertools

# loading topology
traj = md.load(topology)
# print all_trajs
print traj
print traj.topology
sys.stdout.flush()

#--------------------------------------------------------------------
# HMM construction
hidden_states = 3
lag_hmm = 50

print('Building HMM using ' + str(hidden_states) + ' at lag time ' + str(lag_hmm) + ' ...')
sys.stdout.flush()

hmm = pyemma.msm.estimate_hidden_markov_model(dtrajs, hidden_states, lag=lag_hmm)

print('Define core sets using a sharp cutoff...')
sys.stdout.flush()

cores = []
for i in range(hidden_states):
    cores.append(np.where(hmm.metastable_memberships[:, i] > 0.95)[0])

# define a bunch of membership arrays
hmm_membership = hmm.metastable_memberships;

print('Going back from microstates to configurations in such a way that configurations are being associated with a very specific macrostate:')
print('Number of trajectories = ' + str(n_traj))
print('Number of hidden states = ' + str(hidden_states))
sys.stdout.flush()

# cutoff to assign macrostate membership
cutoff = 0.95;
# cutoff to assign transition state membership
close = 0.2

states = [[] for k in range(n_traj)]   # multidimensional list for states
transstates = [[] for k in range(n_traj)]    # multidimensiaonl list for transition states

for i in range(n_traj):
        for k in range(hidden_states):
                states[i].append(np.where(hmm_membership[dtrajs[i][:], k]> cutoff)[0]);   # considering state kth in trajectory ith

for j in range(hidden_states):
        print('percentage associated with macrostate ' + str(j) + ' = ')
    sys.stdout.flush()
        for k in range(n_traj):

                print(np.shape(states[k][j])[0])
                print(np.shape(states[k][j])[0]/float(np.shape(dtrajs[k])[0])*100)
        sys.stdout.flush()

        print('\n')

import itertools

hidden_transstates = list(itertools.combinations(range(hidden_states), 2))
print('number of transition states in system = ' + str(len(hidden_transstates)))
sys.stdout.flush()

for i in range(n_traj):
        for tmp in hidden_transstates:

                transstates[i].append(np.where((hmm_membership[dtrajs[i][:],tmp[0]]>close)*(hmm_membership[dtrajs[i][:],tmp[1]]>close))[0]);

j = 0
for tmp in hidden_transstates:
        print('percentage associated with transistion state ' + str(tmp) + ' = ')
    sys.stdout.flush()
        for k in range(n_traj):

                print(np.shape(transstates[k][j])[0])
                print(np.shape(transstates[k][j])[0]/float(np.shape(dtrajs[k])[0])*100)
        sys.stdout.flush()

        j = j +1
        print('\n')

#--------------------------------------------------------------------
# Difuusion map calculation
what_state='s'
state_idx = 0 # here, the first metastable state is considered

print('Focusing on state number ' + str(state_idx))
sys.stdout.flush()

eps_list = [0.002, 0.003, 0.004, 0.005, 0.006]

print('Considering epsilon values = ' + str(eps_list))
print('\n\n')
sys.stdout.flush()

# Loading trajectory data

import pyemma.coordinates as coor

# define features to load for spacetime diffusion map analysis: heavy atom coordinates only.
print('define basis functions: heavy atom coordinates')
print('\n')
sys.stdout.flush()

featurizer = coor.featurizer(topology)
featurizer.add_selection(featurizer.select_Heavy())

print(featurizer.dimension())
sys.stdout.flush()

# use featurizer to read in trajectory
X1 = coor.load(allfiles1, featurizer, stride=nskip)
# concatenating trajectory chunks into one single trajectory
X1 = np.vstack(X1)

print(X1.shape)

print('trajectory loaded!')
sys.stdout.flush()

# extracting the (indices) subset of configurations from the whole trajectory that was just loaded
state = [[] for k in range(n_traj)]
for i in range(n_traj):
    for j in range(hidden_states):
        state[i].append(states[i][j][np.where(states[i][j]%int(nskip) == 0)]/int(nskip))

my_idx1 = state[0][state_idx]

X1_slice = X1[my_idx1,:]

print(np.shape(X1_slice))

print('Stacking together configurations...')
X_slice = np.vstack((X1_slice))

print('total number of configurations = ' + str(X_slice.shape[0]))
sys.stdout.flush()

# Load Space Time Diffusion Map helper functions

def kernel_neighbor_search(A, r, epsilon, sparse=False):
    """
    Analyzes one frame: uses a nearest neighbor algorithm to compute all distances up to cutoff r, 
    generates the diffusion kernel sparse matrix
    
    Parameters:
        A:   nparray (m, 3), m number of heavy atoms
             array of coordinates of heavy atoms
        r:   scalar, cutoff
             epsilon: scalar, localscale
             
    Return:
        kernel: sparse matrix (m,m)
                diffusion kernel matrix (to be passed on to the SpaceTimeDMap subroutine)
    """

    #calling nearest neighbor search class
    kernel = neigh_search.radius_neighbors_graph(A, r, mode='distance')
    # computing the diffusion kernel value at the non zero matrix entries
    kernel.data = np.exp(-(kernel.data**2)/(epsilon))

    # diagonal needs to be added separately
    kernel = kernel + sps.identity(kernel.shape[0], format = 'csr')

    if sparse:
        return kernel
    else:
        return kernel.toarray()

#--------------------------------------------------------------------

def matrix_B(kernel, sparse = False):

    if sparse:
        m = kernel.shape[0]
        D = sps.csr_matrix.sum(kernel, axis=0)
        Q = sps.spdiags(1./D, 0, m, m)
        S = kernel * Q
        B = (S*(sps.csr_matrix.transpose(S)))/(sps.csr_matrix.sum(S, axis=1))
    else:
        D = np.sum(kernel, axis = 1)
        S = kernel*(1./D)
        B = (np.dot(S, S.T)/(np.sum(S, axis = 1))).T

    return B

#--------------------------------------------------------------------

def compute_SpaceTimeDMap(X, r, epsilon, sparse=False):
    """
    computes the SpaceTime DIffusion Map matrix out of the dataset available
    Parameters
    -------------------
    X: array T x m x 3
      array of T time slices x 3m features (m=number of atoms, features are xyz-coordinates of the atoms)
    r: scalar
      cutoff radius for diffusion kernel
    epsilon: scalar
      scale parameter
     
    Returns
    -------------------
    ll: np.darray(m)
      eigenvalues of the SpaceTime DMap
    u: ndarray(m,m)
      eigenvectors of the SpaceTime DMap. u[:,i] is the ith eigenvector corresponding to i-th eigenvalue
    SptDM: ndarray(m,m)
      SpaceTime Diffusion Matrix, eq (3.13) in the paper, time average of all the matrices in the cell list
    """

    # initialize the Spacetime Diffusion Map matrix 
    # that will be averaged over the different timeslices 
    # and the over the different trajectories 
    m = np.shape(X)[1]
    T = np.shape(X)[0]
    #SptDM =  sps.csr_matrix((m, m)) 
    SptDM = np.zeros((m,m))

    # loop over the indipendent trajectories
    for i_t in range(T):
        if (i_t % 1e4==0):
            print 'time slice ' + str(i_t)
        # selecting the heavy atoms coordinates in the timeslice s
        # compute diffusion kernel using data at timeslice s
        distance_kernel = kernel_neighbor_search(X[i_t,:,:], r, epsilon, sparse=sparse)
        SptDM += matrix_B(distance_kernel, sparse=sparse)

    # divide by the total number of timeslices considered
    # this define the Q operator 
    SptDM /= T

    # Computing eigenvalues and eigenvectors of the SpaceTime DMap
    if sparse:
        ll, u = spl.eigs(SptDM, k = 50, which = 'LR')
        ll, u = sort_by_norm(ll, u)
    else:
        ll, u = np.linalg.eig(SptDM)
        ll, u = sort_by_norm(ll, u)

    return ll, u, SptDM

#--------------------------------------------------------------------

def sort_by_norm(evals, evecs):
    """
    Sorts the eigenvalues and eigenvectors by descending norm of the eigenvalues
    Parameters
    ----------
    evals: ndarray(n)
        eigenvalues
    evecs: ndarray(n,n)
        eigenvectors in a column matrix
    Returns
    -------
    (evals, evecs) : ndarray(m), ndarray(n,m)
        the sorted eigenvalues and eigenvectors
    """
    # norms
    evnorms = np.abs(evals)
    # sort
    I = np.argsort(evnorms)[::-1]
    # permute
    evals2 = evals[I]
    evecs2 = evecs[:, I]
    # done
    return (evals2, evecs2)

#--------------------------------------------------------------------

def reshape_array(X):
    T, m = np.shape(X)
    d = 3
    m = m / d
    A = np.zeros((T, m, d))
    for i_t in range(T):
        A[i_t,:,0] = X[i_t,np.arange(0,3*m,3)];
        A[i_t,:,1] = X[i_t,np.arange(1,3*m,3)];
        A[i_t,:,2] = X[i_t,np.arange(2,3*m,3)];
    return A

#--------------------------------------------------------------------
# Effectively Run spacetime diffusion map analysis
print(' ################         Perform spacetime diffusion map calculations...     ################')
sys.stdout.flush()

X = reshape_array(X_slice)
print(X.shape)
sys.stdout.flush()


eigenvalues = []
eigenvectors = []


if what_state == 's':
        print('Focusing on core states = ' + str(state_idx) + '...')
        filename ='2F4K_' + str(hidden_states) + 'stateHMM_Sptdmap_state' + str(state_idx) + '.npz'
else:
        print('Focusing on transitions state between state ' + str(state_idx) + '...')
        filename = '2F4K_' + str(hidden_states) + 'stateHMM_Sptdmap_trans' + str(state_idx) + '.npz'

print('File where space time diffusoon map results are going to be stored in = ' + str(filename))
sys.stdout.flush()

for eps in eps_list:
    print('epsilon = ' + str(eps))
    sys.stdout.flush()
    r = 2*np.sqrt(eps);
    ll, u, SptDM = compute_SpaceTimeDMap(X, r, eps, sparse=False);
    eigenvalues.append(ll)
    eigenvectors.append(u)

#--------------------------------------------------------------------
# save results
my_dict = {}

my_dict['epsilon'] = eps_list
my_dict['lambdas'] = eigenvalues
my_dict['eigenvectors'] = eigenvectors

np.savez_compressed(filename, **my_dict)
print('Computation and Saving step successful! Data ready to be analyzed')
