#!/usr/bin/env python3
from driver import drive
import os, shutil, json
from os.path import join as pjoin
from time import time
import sys

import numpy as np

from ase.io import read as ase_read

def loop_F(F0, F1, Fstep, update_config=True):
    t0 = time()
    
    # update_config = True # Use last config of a run as start of the next

    # Read params
    params_fname = 'params.json'
    with open(params_fname, 'r') as inj:
        params = json.load(inj)
    # Get force range from command line
    print("Start F0=%.4g end F1=%.4g stpe dF=%.4g" % (F0, F1, Fstep))
    pwd =  os.environ['PWD']
    print('Working in ', pwd)
    #move_fname = ['out.dat', 'traj.dat', 'traj.xyz']
    move_fname = ['out.dat', 'traj.xyz']

    for F in np.arange(F0, F1, Fstep):

        print('--------- ON F=%.8g -----------' % F)
        params['F_ext'] = F
        drive(params)

        cdir = 'F_%.4g' % F
        os.makedirs(cdir, exist_ok=True)

        for cfname in move_fname:
            shutil.move(pjoin(pwd, cfname), pjoin(pwd, cdir, cfname))   
        cfname = params['fname'] 
        shutil.copy(pjoin(pwd, cfname), pjoin(pwd, cdir, cfname))

        params['F_ext'] = float(params['F_ext'])
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
    F0, F1, Fstep = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    update_config = True
    deping_ramp(F0, F1, Fstep, update_config)
    
