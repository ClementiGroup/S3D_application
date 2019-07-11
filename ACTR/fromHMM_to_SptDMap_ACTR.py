# This script is to perform space time diffusion map calculation for each simulation trajectory

#--------------------------------------------------------------------

# INPUT needed:
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
path_results = './'    # where clustering results should be saved
path_full_traj = './traj/'       # where the full trajectory sits

#--------------------------------------------------------------------
# preparing loading trajectories with the featurizer
import mdtraj as md
# parameters
nskip = 1
print('frames to be skipped between successive configurations = ' + str(nskip))
sys.stdout.flush()

# collect all file names in a list...then define featurizer to actually read in trajectories

trajfiles = [path_full_traj + 'ACTR_urea0_all.dcd'] # here, the simulation under 0M urea is used
topology = path_full_traj + 'trj-H_whole_urea0_s1.pdb'

import pyemma.coordinates as coor
import itertools

# loading topology
traj = md.load(topology)
# print all_trajs
print(traj)
print(traj.topology)
sys.stdout.flush()

#--------------------------------------------------------------------
# Difuusion map calculation
eps_list = [0.2, 0.3, 0.4, 0.5, 0.6]

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
X1 = coor.load(trajfiles, featurizer, stride=nskip)

# concatenating trajectory chunks into one single trajectory
X1 = np.vstack(X1)

print(X1.shape)

print('trajectory loaded!')
sys.stdout.flush()

#-------------------------------------------------------------------
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
            print('time slice ' + str(i_t))
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
    m = int(m / d)
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

X = reshape_array(X1)
print(X.shape)
sys.stdout.flush()


eigenvalues = []
eigenvectors = []


filename ='ACTR_0_Sptdmap' + '.npz'

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
