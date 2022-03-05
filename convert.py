import json
import numpy as np
from numpy import pi, sqrt

# Read params
params_fname = 'params-orig.json'
with open(params_fname, 'r') as inj:
    params = json.load(inj)

# Read parameters
print("ORIGINAL Params file")
for k, v in params.items():
    print("%20s :" % k, v)

m = 1
params['m'] = m

# MD params
dt = params['dt']
nstep = params['nstep']

# Substrate
a_s, eps = params['a_s'], params['eps']
# Chain
K, a_c = params['K'], params['a_c']

# External Drivers
F_ext, F_lhs, F_rhs = params['F_ext'], params['F_lhs'], params['F_rhs']

# Langevin
gamma, T = params['gamma'], params['T']

# Boundary conditions
BC = int(params['BC'])
if BC not in [0,1]: raise ValueError("Boundary condition must be 0 for OBC or 1 for PBC not %s" % str(BC))
#li0, li1 = params['li0'], params['li1']

# ------------------- RESCALE TO ADIMENSIONAL VALUES -------------------- #
# From here on all is turned adimensional! Substrate spacing becomes 2pi, energy scale is eps/2
# Create conversion factor dictionary and save it
conv = {
    'm' : m,                                 # mass
    'l' : a_s/(2*pi),                        # length
    'E' : eps/2,                             # energy
    't' : a_s/(2*pi)*sqrt(2*m/eps),          # time
    'freq' : 1/(a_s/(2*pi)*sqrt(2*m/eps)),   # frequency
    'F': 2*pi/a_s*eps/2,                     # force
    'K': eps/2*(2*pi/a_s)**2,                # spring constant
    'v': sqrt(eps/(2*m)),                    # velocity
}
with open('conv.json', 'w') as outj: json.dump(conv, outj, indent=True) # save to json
# Convert input
params['m'] /= conv['m'] # Useless but gives the idea
params['a_c'] /= conv['l']
params['a_s'] /= conv['l']
params['eps'] /= conv['E']
params['dt'] /= conv['t']
params['g'] = K/conv['K']
params['gamma'] /= conv['freq']
params['T'] /= conv['E'] # T=kBT, energy
params['F_ext'] /= conv['F']
params['F_lhs'] /= conv['F']
params['F_rhs'] /= conv['F']
params['m'] /= conv['m'] # USELESS BUT LET'S BE CONSISTENT
params['offset'] /= conv['l']

print("CONVERTED Params file")
for k, v in params.items():
    print("%20s :" % k, v)
with open('params.json', 'w') as outj: json.dump(params, outj, indent=True)
