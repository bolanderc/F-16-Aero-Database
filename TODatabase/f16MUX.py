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

#my_scene.display_wireframe(show_vortices=True)
#my_scene.export_stl(filename="F16.stl")
alpha = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
beta = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
d_e = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
d_a = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
d_r = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
p = np.linspace(-1.57, 1.57, 5)
q = np.linspace(-1.57, 1.57, 5)
r = np.linspace(-0.3925, 0.3925, 5)
cases = list(itertools.product(alpha, beta, d_e, d_a, d_r, p, q, r))
paramlist = []
num_cases_per_core = round(len(cases)/multiprocessing.cpu_count())
i = 0
while i < len(cases):
    paramlist.append(cases[i:i + num_cases_per_core])
    i += num_cases_per_core
def database(params):
    print(multiprocessing.current_process())
    num_cases = len(params)
    input_file = "F16_input.json"
    my_scene = MX.Scene(input_file)
    CL = np.zeros(num_cases)
    CD = np.zeros(num_cases)
    CS = np.zeros(num_cases)
    Cl = np.zeros(num_cases)
    Cm = np.zeros(num_cases)
    Cn = np.zeros(num_cases)
    for i in range(num_cases):
        alpha = params[i][0]
        beta = params[i][1]
        d_e = params[i][2]
        my_scene.set_aircraft_state(state={"alpha" : alpha,
                                           "beta" : beta,
                                           "velocity" : 222.5211})
        my_scene.set_aircraft_control_state(control_state={"elevator" : d_e})
        FM = my_scene.solve_forces()
        CL[i] = FM["F16"]["total"]["CL"]
        CD[i] = FM["F16"]["total"]["CD"]
        CS[i] = FM["F16"]["total"]["CS"]
        Cl[i] = FM["F16"]["total"]["Cl"]
        Cm[i] = FM["F16"]["total"]["Cm"]
        Cn[i] = FM["F16"]["total"]["Cn"]
    return [CL, CD, CS, Cl, Cm, Cn]

#t0 = time.time()
#for i in range(len(paramlist)):
#            print(database(paramlist[i]))
#t1 = time.time()
#print(t1 - t0)
#t0 = time.time()
pool = multiprocessing.Pool()

coeffs = pool.map(database, cases)
pool.close()
pool.join()
#t1 = time.time()
#print(t1 - t0)
