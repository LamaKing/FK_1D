import json, sys
from time import time
from numpy import pi, sqrt, cos, sin
import numpy as np
from scipy.integrate import solve_ivp

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from FK_1D import derivs, sub_en, spring_en
from create_chain import ase2chain, chain2ase


# Start the clock
t0=time()

# Read params
params_fname = 'params.json'
with open(params_fname, 'r') as inj:
    params = json.load(inj)

print("Params file")
for k, v in params.items():
    print("%20s :" % k, v)

# Substrate
eps, a_s = params['eps'], params['a_s']
# Chain
K, a_c = params['K'], params['a_c']
# Integration
F_ext, F_lhs, F_rhs = params['F_ext'], params['F_lhs'], params['F_rhs']
Famp, alpha = eps*pi/a_s, 2*pi/a_s # Shortcuts for derivatives
# Langevin
gamma = params['gamma']

dt = params['dt']

pltflag_v = [0,0,0]
for i, flg in enumerate(sys.argv[1:]):
    pltflag_v[i] = int(flg)
print("Plotting flag", pltflag_v)

if not any(pltflag_v):
    print("No plot flag, exit before loading!")
    exit(0)


traj_fname = 'traj.xyz'
print("Load trajectory at file %s" % traj_fname)
gg_v = ase_read(traj_fname, index=':')

trajall = np.array([x.positions[:,0] for x in gg_v])
print("time x particles", trajall.shape)
Np = trajall.shape[1]

skip_steps = 10

nsteps = trajall.shape[0]
# Missing the skip param!
tvec = dt*np.array(range(nsteps))

#----------------------------------------------------------------------------------------------
# SIMPLE TRAJECTORY
if pltflag_v[0]:
    print("Simple trajectory")
    plt.plot(trajall[::skip_steps,:], tvec[::skip_steps])

    plt.vlines(a_s*np.array(range(-int(1.2*a_c/a_s*Np),int(1.2*a_c/a_s*Np))), *plt.ylim(),
               color='gray', ls=':', lw=0.5)

    plt.xlim([min(trajall.flatten())-5*a_s, max(trajall.flatten())+5*a_s])
    plt.xlabel('$x_i$')
    plt.ylabel('$t$')

    plt.show()
    print('-'*80)

#----------------------------------------------------------------------------------------------
# BOND LENGTH DIFFERENCE
if pltflag_v[1]:
    print("Bond length difference")
    def bond_diff(y):
    # do it with numpy diff
        diff_vec = np.zeros(Np-1)
        for ii in range(0,Np-1):
            d = y[ii+1]-y[ii]
            diff_vec[ii-1] = d - a_c
    #         print(ii+1,'-', ii,':', d)
        return diff_vec

    bd_max = -1e30
    print(nsteps, trajall.shape)
    for ii in range(nsteps):
        cmax = max(np.abs(bond_diff(trajall[ii])))
        if cmax > bd_max: bd_max = cmax
    print("Largest deviation from equilibrium max(|l-l0|/l0)=", bd_max)
    delta = bd_max
    cnorm = Normalize(-delta, delta)

    status_str = "Step %5i t=%8.4g (%5.2f%%)"
    for ii in range(0, nsteps, skip_steps):
        if ii % (nsteps/10) == 0:
            print(status_str % (ii,  ii*dt, ii/nsteps*100))
        plt.scatter((trajall[ii,1:Np]+trajall[ii,0:Np-1])/2, (Np-1)*[tvec[ii]], c=bond_diff(trajall[ii]),
                    marker='o', cmap='RdBu', norm=cnorm, ec='none', lw=0.1, s=5)
                    #marker='o', cmap='RdBu', norm=cnorm, ec='none', lw=0.1, s=0.8)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Bond length')

    plt.vlines(a_s*np.array(range(-int(1.2*a_c/a_s*Np),int(1.2*a_c/a_s*Np))), *plt.ylim(),
               color='gray', ls=':', lw=0.5)
    plt.ylim([0,tvec[-1]])
    #plt.xlim([-30,30])
    plt.xlim([min(trajall.flatten())-5*a_s, max(trajall.flatten())+5*a_s])

    plt.xlabel('$x_i$')
    plt.ylabel('$t$')
    plt.show()
    print('-'*80)

#----------------------------------------------------------------------------------------------
# SUBSTRATE POTENTIAL
if pltflag_v[2]:
    print("Substrate potential")
    cnorm = Normalize(-eps, 0)
    status_str = "Step %5i (%5.2f%%)"

    for ii in range(0, Np):
        if ii % (Np/10) == 0:
            print(status_str % (ii, ii/Np*100))
        plt.scatter(trajall[::skip_steps,ii], tvec[::skip_steps], c=sub_en(trajall[::skip_steps,ii], params),
                    cmap='magma_r', norm=cnorm, ec='none', lw=0.5, s=5)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$E_\mathrm{sub}$')

    plt.vlines(a_s*np.array(range(-int(1.2*a_c/a_s*Np),int(1.2*a_c/a_s*Np))), *plt.ylim(),
               color='gray', ls=':', lw=0.5)
    plt.ylim([0,tvec[-1]])
    #plt.xlim([-30,30])
    plt.xlim([min(trajall.flatten())-5*a_s, max(trajall.flatten())+5*a_s])

    plt.xlabel('$x_i$')
    plt.ylabel('$t$')
    plt.show()
    print('-'*80)
