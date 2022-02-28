import json
import numpy as np
from ase import Atoms

def chain2ase(xvec, vvec, mvec):
    """Convert 1D numpy arrays positions, velocity and masses  to ASE Atoms object"""

    ase_chain = Atoms(positions=np.column_stack((xvec, [0]*len(xvec), [0]*len(xvec))),
                      masses=mvec
                      )
    ase_chain.set_velocities(np.column_stack((vvec, [0]*len(xvec), [0]*len(xvec))))
    return ase_chain


def ase2chain(ase_chain):
    """Convert ASE Atoms object to 1D positions, velocity and masses NumPy arrays"""

    xvec = ase_chain.positions[:,0]
    vvec = ase_chain.get_velocities()[:,0]
    mvec = ase_chain.get_masses()

    return xvec, vvec, mvec

params_fname = 'params.json'
with open(params_fname, 'r') as inj:
    params = json.load(inj)

Np = params['Np'] # Number of atoms
a_c = params['a_c'] # Spacing

offset =  params['offset'] # Offset for chain

# Position
xvec = offset + a_c*np.array(range(-int(Np/2), int(Np/2)))

# Velocities
v0 = 0
vvec =  np.array(len(xvec)*[v0])

# Masses
mass = 1
mvec = np.array(len(xvec)*[mass])


fname = 'test.xyz'
chain2ase(xvec, vvec, mvec).write(fname)
