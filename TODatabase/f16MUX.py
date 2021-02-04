"""
The purpose of this script is to demonstrate the functionality of MachUpX for modelling
a traditional (straight main wing, simple empennage) airplane. Run this script using:

$ python traditional_example.py

The input file is also written such that the same analyses will be performed if run using
the `python -m` command. This is done using:

$ python -m machupX traditional_input.json
"""

# Import the MachUpX module
import machupX as MX
import matplotlib.pyplot as plt
import numpy as np
import stdatmos as atmos
import scipy.optimize as optimize
import time
import itertools
import multiprocessing

# Input the json module to make displaying dictionaries more friendly
import json


input_file = "F16_input.json"

my_scene = MX.Scene(input_file)

#my_scene.display_wireframe(show_vortices=True)
#my_scene.export_stl(filename="F16.stl")
alpha = np.linspace(-np.deg2rad(30), np.deg2rad(30), 3)
beta = np.linspace(-np.deg2rad(30), np.deg2rad(30), 3)
d_e = np.linspace(-np.deg2rad(30), np.deg2rad(30), 3)
coeffs = np.zeros((6, len(alpha), len(beta), len(d_e)))
paramlist = list(itertools.product(alpha, beta, d_e))
def database(params):
    alpha = params[0]
    beta = params[1]
    d_e = params[2]
    my_scene.set_aircraft_state(state={"alpha" : alpha,
                                       "beta" : beta,
                                       "elevator" : d_e,
                                       "velocity" : 222.5211})
    FM = my_scene.solve_forces(dimensional=False)
    CL = FM["F16"]["total"]["CL"]
    CD = FM["F16"]["total"]["CD"]
    CS = FM["F16"]["total"]["CS"]
    Cl = FM["F16"]["total"]["Cl"]
    Cm = FM["F16"]["total"]["Cm"]
    Cn = FM["F16"]["total"]["Cn"]
    return [CL, CD, CS, Cl, Cm, Cn]

#t0 = time.time()
#for i in range(len(alpha)):
#    for j in range(len(beta)):
#        for k in range(len(d_e)):
#            coeffs[:, i, j, k] = database([alpha[i], beta[j], d_e[k]])
#t1 = time.time()
#print(t1 - t0)
t0 = time.time()
pool = multiprocessing.Pool()

coeffs = pool.map(database, paramlist)
t1 = time.time()
print(t1 - t0)
