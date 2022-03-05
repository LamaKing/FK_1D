#!/usr/bin/env python3

import numpy as np
from numpy import pi
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
import matplotlib.pyplot as plt
import json

print("Loading traj")
gg_traj = ase_read('traj.xyz', index=':')
# Read params
params_fname = 'params.json'
with open(params_fname, 'r') as inj:
    params = json.load(inj)

a_c, Np = params['a_c'], len(gg_traj[0])
phi = -pi/2
L = a_c*Np
dR = L/20

print("phi=%.4g dR=%.4g" % (phi, dR))
gg_ringtraj = []

def chain2ase(xvec, vvec, mvec):
    """Convert 1D numpy arrays positions, velocity and masses  to ASE Atoms object"""

    ase_chain = Atoms(positions=np.column_stack((xvec, [0]*len(xvec), [0]*len(xvec))),
                      masses=mvec
                      )
    ase_chain.set_velocities(np.column_stack((vvec, [0]*len(xvec), [0]*len(xvec))))
    return ase_chain

def f(x, L):
    """Map the linear coordinate to theta in 0 2pi"""
    return 2*pi*x/L

def g(t, R, phi=0):
    """Map theta to a circle of radius R

    The radius could be a constant or vary for each particle"""

    xx = R*np.cos(t+phi)
    yy = R*np.sin(t+phi)
    return np.array([xx, yy]).T

ii = 0
Nframe = len(gg_traj)
for gg in gg_traj:
    if ii % int(Nframe/10) == 0: print("On frame %i of %i (%.2f%%)" % (ii, Nframe, ii/Nframe*100))

    xx = gg.positions[:,0]
    xx_ring = g(f(xx, L), L+dR*gg.positions[:,1], phi)

    c_ase = Atoms(positions=np.column_stack((xx_ring[:,0], xx_ring[:,1], gg.positions[:,1])),
                  velocities=gg.get_velocities(),
                  masses=params['m']*np.ones(xx_ring.shape[0])
                  )
    gg_ringtraj.append(c_ase)
    ii += 1

trajfname = 'traj-ring.xyz'
print("Save to %s" % trajfname)
ase_write(trajfname, gg_ringtraj)
