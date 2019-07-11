# This script is for TICA calculation, which contains 2 parts:
# first, we choose differernt valuses of lag time, and check the convergence of TICA by calculating the TICA timescales
# then, with a reasonable lag time (the minimal possible value where the timescales are well converged), 
# we project the input trajectory onto the TICA space, which will output TICA coordinates

#--------------------------------------------------------------------

# INPUT needed:
        # the simulation trajectory and topology file
        # sequence of lag time(s)

# OUTPUT: list of TICA timescales for different lag time 
#         or TICA timescales, eigenvalues and coordinates for a spesific lag time 

#--------------------------------------------------------------------

# coding : utf-8

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
import sys
import math
import mdtraj as md
import pyemma
print(pyemma.__version__)

#---------------------------------------------------------------------
# parameter setting

nskip = 1
print('frames to be skipped between successive configurations = ' + str(nskip))
delta = 100.0/(500000.0/nskip)    # dt [in microseconds]
print('timestep between successive configurations             = ' + str(delta))
tica_dim = 50
print('number of tica dimensions set by user                  = ' + str(tica_dim))
n_traj = 1
print('number of traj chunks per trajectory (DE Shaw)         = ' + str(n_traj))
sys.stdout.flush()

# For TICA convergence check
lags = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

# For TICA coordinates extraction
#lags = [100]

print('lag time(s) for TICA calculation = ' + str(lags))
sys.stdout.flush()

#--------------------------------------------------------------------
# define function reverse distance

def one_over_d(traj):
    dd = 1./(md.compute_contacts(traj,contacts='all',scheme='closest-heavy')[0])
    return dd

#--------------------------------------------------------------------
# define input files path

trajfile = './traj/2F4K-0-protein_all.dcd'
topology = './traj/2F4K-0-protein_fixed_noH.pdb'

#--------------------------------------------------------------------
# define features to be used to input to TICA calculation

import pyemma.coordinates as coor
import itertools

traj = md.load(topology)
print('trajectory objects = ' + str(traj))
print('topology object    = ' + str(traj.topology))
sys.stdout.flush()

# define a featurizer
feat = coor.featurizer(topology)

# define basis functions: heavy-atom contact distances, heavy atom coordinates, all torsions
print('define basis functions: heavy-atom contact distances, heavy atom coordinates, all torsions, inverse distances')
print('\n')
sys.stdout.flush()

featurizer = coor.featurizer(topology)
featurizer.add_residue_mindist(residue_pairs='all', scheme='closest-heavy')
featurizer.add_all()
featurizer.add_backbone_torsions(cossin=True)
featurizer.add_chi1_torsions(cossin=True)
indx = md.compute_chi2(traj)[0]
featurizer.add_dihedrals(indx, cossin=True)
indx = md.compute_chi3(traj)[0]
featurizer.add_dihedrals(indx, cossin=True)
indx = md.compute_chi4(traj)[0]
featurizer.add_dihedrals(indx, cossin=True)
indx = md.compute_omega(traj)[0]
featurizer.add_dihedrals(indx, cossin=True)
indx = md.compute_contacts(traj,contacts='all',scheme='closest-heavy')[1]
featurizer.add_custom_func(one_over_d, np.shape(indx)[0])

print(featurizer.describe())
print(featurizer.dimension())
sys.stdout.flush()

# use featurizer to read in trajectories
inp = coor.source([trajfile], featurizer, chunk_size=10000, stride=nskip)

#--------------------------------------------------------------------
# define lag time(s) and run TICA calculation on the trajectory

setup= len(lags)

if setup == 0:
    print('Whoops...something is wrong, it seems like no lag time was provided!')
    sys.stdout.flush()

if setup == 1:
    print('Running TICA calculation using lag time equal to ' + str(lags[0]))
    sys.stdout.flush()

        tica_obj = coor.tica(inp, dim = tica_dim, lag=lags[0], kinetic_map=False, commute_map=True)

        # extract single lag time features
        tica_timescales = tica_obj.timescales
        tica_eigenvalues = tica_obj.eigenvalues
        Y = tica_obj.get_output()

        # save data
        print('Saving TICA results...')
        sys.stdout.flush()

        np.savez_compressed('2F4K_TICA_coordinates_lag'+str(lags[0])+'.npz', Y = Y)
        np.savez_compressed('2F4K_TICA_timescales_lag'+str(lags[0])+'.npz', tica_timescales = tica_timescales)
        np.savez_compressed('2F4K_TICA_eigenvalues_lag'+str(lags[0])+'.npz', tica_eigenvalues = tica_eigenvalues)

if setup > 1:
    print('Running multiple lag time TICA calculations to check for convergence...')
    sys.stdout.flush()

    objs = [coor.tica(inp, dim = tica_dim, lag=tau, kinetic_map=False, commute_map=True) for tau in lags]

    # extracting timescales from eigenvalues
    its = [obj.timescales for obj in objs]

    # stacking timescales for different lag times
    ndim = np.shape(its[0])[0]
    print('number of dimension', ndim)
    nlags = len(lags)
    print('number of lag times', nlags)
    timescale = np.zeros((nlags, ndim))
    sys.stdout.flush()

    # storing timescales
    for i in range(len(lags)):
        timescale[i,:] = its[i]

        # save data
        filename = '2F4K_TICA_timescales_all.npz'

        my_dict = {}
        my_dict['timescales'] = timescale
        my_dict['lags'] = lags

        np.savez_compressed(filename, **my_dict)

print('TICA calculation went to completion all right!')
