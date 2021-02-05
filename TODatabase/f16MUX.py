"""
The purpose of this script is to demonstrate the functionality of MachUpX for modelling
a traditional (straight main wing, simple empennage) airplane. Run this script using:

$ python traditional_example.py

The input file is also written such that the same analyses will be performed if run using
the `python -m` command. This is done using:

$ python -m machupX traditional_input.json
"""
from multiprocessing import set_start_method
set_start_method("spawn")

# Import the MachUpX module
import machupX as MX
import matplotlib.pyplot as plt
import numpy as np
import stdatmos as atmos
import scipy.optimize as optimize
import time
import itertools
import multiprocessing
from multiprocessing import get_context

# Input the json module to make displaying dictionaries more friendly
import json

#my_scene.display_wireframe(show_vortices=True)
#my_scene.export_stl(filename="F16.stl")
alpha = np.linspace(-np.deg2rad(30), np.deg2rad(30), 3)
beta = np.linspace(-np.deg2rad(30), np.deg2rad(30), 3)
d_e = np.linspace(-np.deg2rad(30), np.deg2rad(30), 3)
cases = list(itertools.product(alpha, beta, d_e))
paramlist = []
num_cases_per_core = round(len(cases)/8)
i = 0
while i < len(cases):
    paramlist.append(cases[i:i + num_cases_per_core])
    i += num_cases_per_core
def database(params):
    with get_context("spawn").Pool() as pool:
        print(multiprocessing.current_process())
        input_file = "F16_input.json"
        my_scene = MX.Scene(input_file)
        num_cases = len(params)
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

# t0 = time.time()
# for i in range(len(paramlist)):
#             print(database(paramlist[i]))
# t1 = time.time()
# print(t1 - t0)
t0 = time.time()
pool = multiprocessing.Pool(processes=len(paramlist))

coeffs = pool.map(database, paramlist)
pool.close()
pool.join()
t1 = time.time()
print(t1 - t0)
