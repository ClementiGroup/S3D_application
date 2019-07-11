# This script is to build Markov State Model out of TICA coordinates


#--------------------------------------------------------------------

# INPUT needed:
        # TICA calculation results (eigenvalues, eigenvectors, timescales)
        # neigen, number of TICA coordinates to be selected
        # sequence of lag times
        # number of clusters to be considered

# OUTPUT: discrete trajectory obtained with k-means clustering
#         list of transition matrices for different lag times

#--------------------------------------------------------------------


import numpy as np
import os
import sys
import mdtraj as md
import pyemma
import pyemma.coordinates as coor
import msmtools
import math

print('trajectory time step = ',0.2,'ns')
sys.stdout.flush()

#--------------------------------------------------------------------
# read in all TICA coordinates

print('read all TICA coordinates and timescales')
sys.stdout.flush()

n_traj = 1
print('number of independent trajectories =' + str(n_traj))
sys.stdout.flush()

npzfile = np.load('/scratch/wy14/S3D/2F4K/TICA/2F4K_TICA_coordinates_lag100.npz')
npzfile.keys()
Y = npzfile['Y']
print(np.shape(Y))
sys.stdout.flush()

npzfile = np.load('/scratch/wy14/S3D/2F4K/TICA/2F4K_TICA_timescales_lag100.npz')
npzfile.keys()
tica_timescales = npzfile['tica_timescales']
print(np.shape(tica_timescales))
sys.stdout.flush()

npzfile = np.load('/scratch/wy14/S3D/2F4K/TICA/2F4K_TICA_eigenvalues_lag100.npz')
npzfile.keys()
tica_eigenvalues = npzfile['tica_eigenvalues']
print(np.shape(tica_eigenvalues))
sys.stdout.flush()

# choose number of TICA coordinates to be considered
neigen = 10

#--------------------------------------------------------------------
# choose TICA coordinates
print('choose first' + str(neigen) + ' TICA coordinates')
sys.stdout.flush()

Ys = []
for j in range(n_traj):
    Ys_tmp = np.zeros((np.shape(Y[j])[0], neigen))
    for i in range(neigen):
        Ys_tmp[:,i] = Y[j][:,i]
    Ys.append(Ys_tmp)

#--------------------------------------------------------------------
# Use TICA coordinates as clustering coordinates

print('clustering')
sys.stdout.flush()

MSMlags = np.array([1])
for i in range(1,4,1):
    nmin = 10**(i)
    nmax = 10**(i+1)
    dn = 10**(i)
    MSMlags1 = np.arange(nmin,nmax,dn)
    MSMlags = np.concatenate([MSMlags,MSMlags1])

nlagsMSM = np.shape(MSMlags)[0]
print('number of different lag times chosen = ' + str(nlagsMSM))
print('lag time values used = ' + str(MSMlags))
sys.stdout.flush()

n_clusters = 1000
clustering = coor.cluster_kmeans(list(Ys),k=n_clusters,max_iter=10000)
dtrajs = clustering.dtrajs

my_dict = {}

my_dict['n_clusters'] = n_clusters
my_dict['micro_membership'] = dtrajs
my_dict['centers'] = clustering.clustercenters

np.savez_compressed('2F4K_MSM_10TICA_clusters_1000.npz', **my_dict)

#--------------------------------------------------------------------
# Build Markov State Model out of clustered data and save Markov transition matrices to file

print('Building Markov Model at different lag times...')
sys.stdout.flush()

T = []
for j in range(nlagsMSM):
    print('building markov state model at lag time = ' + str(MSMlags[j]) + ' steps...')
    sys.stdout.flush()
    MSM = pyemma.msm.estimate_markov_model(dtrajs,lag=MSMlags[j])
    T.append(MSM.transition_matrix)

my_dict = {}
count = 0
for j in range(nlagsMSM):
    my_dict[str(j)] = T[j]

np.savez_compressed('2F4K_MSMtransition_matrix_10TICA_1000.npz', **my_dict)

print('Markov Model calculation went to completion, saving OK!')
