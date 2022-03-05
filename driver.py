#!/usr/bin/env python3
import json
from time import time
import numpy as np
from numpy import pi, sqrt, cos, sin
from numpy.random import normal
#from scipy.integrate import solve_ivp, RK45
from RK45_lang import RK45_lang

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from FK_1D import derivs, sub_en, spring_en
from create_chain import ase2chain, chain2ase

# Nope in python they are only global to the module, not imported...
#global eps, a_s, K, a_c, F_ex, F_lhs, F_rhs, Fam, alpha, gamma, T, brand, BC, dt

def drive(params):

    # Start the clock
    t0=time()

    # Read parameters
    print("Params file")
    for k, v in params.items():
        print("%20s :" % k, v)

    name='' # Not usually needed, use folders
    if 'name' in params.keys(): name = params['name']

    # Load sys
    ase_chain = ase_read(params['fname'])
    xvec, vvec, mvec = ase2chain(ase_chain) # We read the vector mass but then we ignore it, all the same
    m, Np = mvec[0], len(xvec)
    print("Loaded %s:" % (params['fname']), ase_chain)
    if Np == 1: raise ValueError('Sorry, need at least two particle to connect springs')
    # MD integration params
    dt = params['dt']
    nstep = params['nstep']
    nskip = params['nskip']

    # Substrate
    a_s, eps = 2*pi, 2
    # Chain
    g, a_c = params['g'], params['a_c']

    # External Drivers
    F_ext, F_lhs, F_rhs = params['F_ext'], params['F_lhs'], params['F_rhs']

    # Langevin
    gamma, T = params['gamma'], params['T']
    gamma0, T0, li0 = params['gamma0'], params['T0'], params['li0']
    gamma1, T1, li1 = params['gamma1'], params['T1'], params['li1']

    # Boundary conditions
    BC = int(params['BC'])
    if BC not in [0,1]: raise ValueError("Boundary condition must be 0 for OBC or 1 for PBC not %s" % str(BC))
    #li0, li1 = params['li0'], params['li1']

    # ------------------- SYSTEM ARRAYS -------------------- #
    # Setup particle-wise Langevin
    gammav = gamma*np.ones(Np)
    brand = np.sqrt(2*T*m*gamma/dt)
    brandv = brand*np.ones(Np)

    brand0 = np.sqrt(2*T0*m*gamma0/dt)
    gammav[:li0] = gamma0
    gammav[Np-li0:] = gamma0
    brandv[:li0] = brand0
    brandv[Np-li0:] = brand0

    Nmid = int(Np/2)
    brand1 = np.sqrt(2*T1*m*gamma0/dt)
    gammav[Nmid-li1:Nmid+li1] = gamma1
    brandv[Nmid-li1:Nmid+li1] = brand1

    print("base", gamma, T, brand)
    print("Split dT PBC")
    print("region 0: size %i" % li0, "from %i to %i and from %i to %i" % (0, li0, Np-li0, Np), gamma0, T0, brand0)
    print("region 1: size %i" % li1, "from %i to %i" % (Nmid-li1, Nmid+li1), gamma1, T1, brand1)

    #print(gammav)
    #print(brandv)

    # Parameters for the FK solver (save to json)
    dparams = {'a_c': a_c, 'g': g,
               'F_ext': F_ext, 'F_lhs': F_lhs, 'F_rhs': F_rhs,
               'brandv': list(brandv), 'gammav': list(gammav),
               'BC': BC
               }
    with open('dparam.json', 'w') as outj: json.dump(dparams, outj)

    # Setup the equation array, combine position and velocity
    eqvec = np.concatenate((xvec, vvec))  # First half is position second half is velocity
    neq = len(eqvec)
    neq2 = int(neq/2) # useful shortcut for division between pos and velox

    # ------------------- PRINT SYSTEM INFO -------------------- #
    # Substrate time scale (adimensional variable!)
    ws, gamma_sc = 1, 2
    # Spring time scale
    wc, gamma_cc  = sqrt(g), 2*sqrt(g)
    header = ' SYSTEM INFO '
    print('-'*20 + header + '-'*20)
    print('Chain: N=%i a_c=%.4g g=%.4g m=%.4g' % (Np, a_c, g, m))
    print('Sub: a_s=%.4g eps=%.4g' % (a_s, eps))
    print('Damping: gamma=%.4g (gamma/gamma_sc=%.4g gamma/gamma_cc=%.4g)' % (gamma, gamma/gamma_sc, gamma/gamma_cc))
    print('Temperature: T=%.4g rand amp=%.4g' % (T, brand))
    print('F_ext=%.4g F_lhs=%.4g F_rhs=%.4g' % (F_ext, F_lhs, F_rhs))
    print('Integrate until tf=%.4g (N=%i) with dt=%.4g (interal nstep=%i)' % (nstep*dt, nstep, dt, nskip))
    print('-'*20 + '-'*len(header) + '-'*20)
    # ---------------------------------------------------------- #

    #-------- OUTPUT SETUP -----------
    trajfname = 'traj%s.xyz' % name
    trajstream = open(trajfname, 'w')
    outfname = 'out%s.dat' % name
    outstream = open(outfname, 'w')

    # !! Labels and print_status data structures must be coherent !!
    num_space, indlab_space= 30, 2  # Width printed numerical values and Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['sub_en', 'spring_en', 'ekin', 'kBT', 'pos_cm[0]', 'Vcm[0]']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='t', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                         for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)
    # Inner-scope shortcut for printing
    def print_status(data):
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space) for val in data]), file=outstream)
    # Update status string
    status_str = "%8i) t=%12.4g (%6.2f%%) xcm=%20.10g (%8.2g) vcm=%20.10g kBT=%20.10g etot=%30.16g"

    #----------------- MD --------------------
    #### INIT
    t, tf = 0, dt*nstep
    it = 0
    # Save first step
    c_sub_en, c_spring_en, c_ekin = np.sum(sub_en(xvec)), np.sum(spring_en(xvec, g, a_c, BC)), np.sum(1/2*vvec**2)
    c_kBT = 2*c_ekin/Np
    xcm, vcm = np.average(xvec), np.average(vvec)

    print(status_str % (it, t, t/tf*100, xcm, xcm/2/pi, vcm , c_kBT, c_sub_en+c_spring_en+c_ekin))
    print_status([t, c_sub_en, c_spring_en, c_ekin, c_kBT, xcm, vcm])

    c_ase = chain2ase(xvec, vvec, mvec)
    c_ase.positions[:,1] = sub_en(xvec) # Use y coord as substrate energy
    c_ase.positions[:,2] = spring_en(xvec, g, a_c, BC) # Use z coord as spring energy
    ase_write(trajfname, c_ase, append=True) # XYZ should be able to do append, careful if you change it!

    # Solve equations with RK45 FIX STEP for Langevin. See module here and SciPy.
    # Wrap parameters for the solver
    fun = lambda t, y: derivs(t, y, dparams)
    solver = RK45_lang(fun, t, eqvec, tf)
    solver.set_step(dt) # Set FIXED time step of intergration
    solver.set_nskip(nskip) # Set number of step done inside the function, not reported

    while t<tf:
        solver.step() # Advance one step
        eqvec = solver.y
        t = solver.t

        # Print status update
        if it % int(nstep/nskip/10) == 0:
            xvec, vvec = eqvec[:neq2], eqvec[neq2:]
            c_sub_en, c_spring_en, c_ekin = np.sum(sub_en(xvec)), np.sum(spring_en(xvec, g, a_c, BC)), np.sum(1/2*vvec**2)
            c_kBT = 2*c_ekin/Np
            xcm, vcm = np.average(xvec), np.average(vvec)
            print(status_str % (it, t, t/tf*100, xcm, xcm/2/pi, vcm , c_kBT, c_sub_en+c_spring_en+c_ekin))

        # Save results
        xvec, vvec = eqvec[:neq2], eqvec[neq2:]
        c_sub_en, c_spring_en, c_ekin = np.sum(sub_en(xvec)), np.sum(spring_en(xvec, g, a_c, BC)), np.sum(1/2*vvec**2)
        c_kBT = 2*c_ekin/Np
        xcm, vcm = np.average(xvec), np.average(vvec)

        print_status([t, c_sub_en, c_spring_en, c_ekin, c_kBT, xcm, vcm])

        c_ase = chain2ase(xvec, vvec, mvec)
        c_ase.positions[:,1] = sub_en(xvec) # Use y coord as substrate energy
        c_ase.positions[:,2] = spring_en(xvec, g, a_c, BC) # Use z coord as spring energy
        ase_write(trajfname, c_ase, append=True) # XYZ should be able to do append, careful if you change it!

        it+=1
    #-------------------------------------------------------

    # Save LAST STEP
    xvec, vvec = eqvec[:neq2], eqvec[neq2:]
    c_sub_en, c_spring_en, c_ekin = np.sum(sub_en(xvec)), np.sum(spring_en(xvec, g, a_c, BC)), np.sum(1/2*vvec**2)
    c_kBT = 2*c_ekin/Np
    xcm, vcm = np.average(xvec), np.average(vvec)

    print(status_str % (it, t, t/tf*100, xcm, xcm/2/pi, vcm , c_kBT, c_sub_en+c_spring_en+c_ekin))
    print_status([t, c_sub_en, c_spring_en, c_ekin, c_kBT, xcm, vcm])

    c_ase = chain2ase(xvec, vvec, mvec)
    c_ase.positions[:,1] = sub_en(xvec) # Use y coord as substrate energy
    c_ase.positions[:,2] = spring_en(xvec, g, a_c, BC) # Use z coord as spring energy
    ase_write(trajfname, c_ase, append=True) # XYZ should be able to do append, careful if you change it!
    trajstream.close()

    # DONE
    print("Finished after %i solver calls" % solver.ncalls,  "Solver status", solver.status)
    t1=time()
    print('Done in %is (%.2fmin)' % (t1-t0, (t1-t0)/60))

if __name__ == "__main__":
    # Read params
    params_fname = 'params.json'
    with open(params_fname, 'r') as inj:
        params = json.load(inj)
    drive(params)
