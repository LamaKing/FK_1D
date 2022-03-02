import json
from time import time
import numpy as np
from numpy import pi, sqrt, cos, sin
from numpy.random import normal
from scipy.integrate import solve_ivp

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from FK_1D import derivs, sub_en, spring_en
from create_chain import ase2chain, chain2ase

def drive(params):

    # Start the clock
    t0=time()

    print("Params file")
    for k, v in params.items():
        print("%20s :" % k, v)

    name=''
    if 'name' in params.keys():
        name = params['name']

    # Substrate
    eps, a_s = params['eps'], params['a_s']
    # Chain
    K, a_c = params['K'], params['a_c']
    # Integration
    F_ext, F_lhs, F_rhs = params['F_ext'], params['F_lhs'], params['F_rhs']
    Famp, alpha = eps*pi/a_s, 2*pi/a_s # Shortcuts for derivatives
    # Langevin
    gamma, T = params['gamma'], params['T']
    brand = np.sqrt(2*T*gamma)
    params['brand'] = brand

    dt = params['dt']
    nstep = params['nstep']

    # Load sys
    ase_chain = ase_read(params['fname'])
    xvec, vvec, mvec = ase2chain(ase_chain)
    Np = len(xvec)
    print(Np, ase_chain)

    # We read the mass but then we ignore it fix it to one
    # Setup the equation of position and velocity
    eqvec = np.concatenate((xvec, vvec))

    ## SYSTEM INFO
    g = K*a_s**2/(2*eps*pi**2) # Coupling
    # Substrate scale
    ws = pi/a_s*sqrt(2*eps)
    gammasc = 2*ws
    # Spring scale
    w0 = sqrt(K) # mass=1
    gamma0c = 2*w0

    print('Lat sub %.4g lat col %.4g. Mismatch a_sub/a_chain=%.4g Np=%i' % (a_s, a_c, a_s/a_c, Np))
    print("Approx natural frequncy of chain w0=%.4g and critical damping gamma0c=%.4g" % (ws, gammasc))
    print("Current damping gamma=%.4g. Ratio gamma/gammasc=%.4g" % (gamma, gamma/gammasc))

    print("Natural frequncy of chain w0=%.4g and critical damping gamma0c=%.4g" % (w0, gamma0c))
    print("Current damping gamma=%.4g. Ratio gamma/gamma0c=%.4g" % (gamma, gamma/gamma0c))

    print("Dimensionless ratio k/ (2 U0 (pi/a_substrate)^2) =", g)

    print("Temperature %.4g. kBT/E_sub=? kBT/E_spring=?" % T)

    # First half is position second half is velocity
    neq = len(eqvec)
    neq2 = int(neq/2) # useful shortcut for division between pos and velox

    # PREPARE VECTOR TRAJECTORY
    cmtraj = [[np.average(eqvec[:neq2]), np.average(eqvec[neq2:])]]
    traj = [eqvec]
    tvec = [0]
    suben_traj = [np.sum(sub_en(eqvec[:neq2], params))]
    springen_traj = [np.sum(spring_en(eqvec[:neq2], params))]
    ase_traj = []

    #-------- OUTPUT SETUP -----------
    trajfname = 'traj%s.xyz' % name
    outfname = 'out%s.dat' % name
    outstream = open(outfname, 'w')

    # !! Labels and print_status data structures must be coherent !!
    num_space = 30 # Width printed numerical values
    indlab_space = 2 # Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['sub_en', 'spring_en', 'ekin', 'pos_cm[0]', 'Vcm[0]']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='dt*it', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                         for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)

    # Inner-scope shortcut for printing
    def print_status(data):
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                       for val in data]), file=outstream)

    # MD param
    t = 0
    print_skip = 10
    # DO MD
    status_str = "Step %5i t=%8.4g (%5.2f%%) xcm=%15.4g vcm=%15.4g etot=%12.5g"
    print('-'*30, 'Drive N=%i' % nstep, '-'*30)
    for it in range(0, nstep):
        eqvec = traj[-1]

        # Solve equations with RK45 see Scipy doc
        sol = solve_ivp(derivs, [t, t+dt], eqvec, args=[params])
        # Can we add bath kick at each time, after solving the equation?
        noise = normal(0, 1, size=neq2) # Gaussian numbers
        sol.y[neq2:, -1] += brand*noise # Add noise to velocities
        t += dt

        tvec.append(sol.t[-1])
        traj.append(sol.y[:, -1])
        xvec, vvec = sol.y[:neq2, -1], sol.y[neq2:, -1]
        c_sub_en, c_spring_en, c_ekin = np.sum(sub_en(xvec, params)), np.sum(spring_en(xvec, params)), np.sum(1/2*vvec**2)
        xcm, vcm = np.average(xvec), np.average(vvec)
        cmtraj.append([xcm, vcm])
        suben_traj.append(c_sub_en)
        springen_traj.append(c_spring_en)

        # Print step results
        if it % print_skip == 0:
            print_status([t, c_sub_en, c_spring_en, c_ekin, xcm, vcm])
            c_ase = chain2ase(xvec, vvec, mvec)
            c_ase.positions[:,1] = sub_en(xvec, params) # Use y coord as substrate energy
            c_ase.positions[:,2] = np.concatenate(([0], spring_en(xvec, params))) # Use z coord as spring energy
            ase_traj.append(c_ase)

        if it % (nstep/10) == 0:
            print(status_str % (it,  it*dt, it/nstep*100, np.average(sol.y[:neq2, -1]), np.average(sol.y[neq2:, -1]), c_sub_en+c_spring_en+c_ekin))

    print(status_str % (it,  it*dt, it/nstep*100,
                        np.average(sol.y[:neq2, -1]),
                        np.average(sol.y[neq2:, -1]),
                        c_sub_en+c_spring_en+c_ekin))

    print_status([t, c_sub_en, c_spring_en, c_ekin, xcm, vcm])
    c_ase = chain2ase(xvec, vvec, mvec)
    c_ase.positions[:,1] = sub_en(xvec, params) # Use y coord as substrate energy
    c_ase.positions[:,2] = np.concatenate(([0], spring_en(xvec, params))) # Use z coord as spring energy
    ase_traj.append(c_ase)

    ase_write(trajfname, ase_traj)

    # Big full-output files
    traj = np.array(traj)
    #cmtraj = np.array(cmtraj)
    #suben_traj = np.array(suben_traj)
    #springen_traj = np.array(springen_traj)

    np.savetxt('traj%s.dat' % name, traj)

    t1=time()
    print('Done in %is (%.2fmin)' % (t1-t0, (t1-t0)/60))

if __name__ == "__main__":
    # Read params
    params_fname = 'params.json'
    with open(params_fname, 'r') as inj:
        params = json.load(inj)
    drive(params)
