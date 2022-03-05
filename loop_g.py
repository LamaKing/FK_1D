#!/usr/bin/env python3
from driver import drive
import os, shutil, json
from os.path import join as pjoin
from time import time
import sys

import numpy as np

from ase.io import read as ase_read

def g_loop(g0, g1, dg, update_config=False):
    t0 = time()
    
    # update_config = True # Use last config of a run as start of the next

    # Read params
    params_fname = 'params.json'
    with open(params_fname, 'r') as inj:
        params = json.load(inj)
    # Get force range from command line
    print("Start g0=%.4g end g1=%.4g stpe dg=%.4g" % (g0, g1, dg))
    pwd =  os.environ['PWD']
    print('Working in ', pwd)
    move_fname = ['out.dat', 'traj.xyz']

    for g in np.arange(g0, g1, dg):

        print('--------- ON g=%.8g -----------' % g)
        params['F_ext'] = 0
        params['F_lhs'] = 0
        params['F_rhs'] = 0
        params['g'] = g
        drive(params)

        cdir = 'g_%.4g' % g
        os.makedirs(cdir, exist_ok=True)

        for cfname in move_fname:
            shutil.move(pjoin(pwd, cfname), pjoin(pwd, cdir, cfname))   
        cfname = params['fname'] 
        shutil.copy(pjoin(pwd, cfname), pjoin(pwd, cdir, cfname))

        params['g'] = float(params['g'])
        #params['brand'] = float(params['brand'])
        with open(pjoin(pwd, cdir, 'params.json'), 'w') as outj:
            json.dump(params, outj)

        if update_config:
            ase_read(pjoin(pwd, cdir, 'traj.xyz'), index=-1).write('start.xyz')
            params['fname'] = 'start.xyz'
        print('-' * 80, '\n')

    t1=time()
    print('Done in %is (%.2fmin)' % (t1-t0, (t1-t0)/60))
if __name__ == "__main__":
    g0, g1, dg = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    update_config = False
    g_loop(g0, g1, dg, update_config)
    
