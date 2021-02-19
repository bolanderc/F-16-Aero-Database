import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from machupX import Scene
import matplotlib.pyplot as plt
import numpy as np
import stdatmos as atmos
import scipy.optimize as optimize
import time
import itertools
from multiprocessing import Pool

# Input the json module to make displaying dictionaries more friendly
import json
input_file = "F16_input.json"
my_scene = Scene(input_file)
# my_scene.display_wireframe(show_vortices=True)
# my_scene.export_stl(filename="F16.stl")
alpha = np.linspace(-np.deg2rad(30), np.deg2rad(30), 2)
beta = np.linspace(-np.deg2rad(30), np.deg2rad(30), 2)
d_e = np.linspace(-np.deg2rad(30), np.deg2rad(30), 2)
d_a = np.linspace(-np.deg2rad(30), np.deg2rad(30), 2)
d_r = np.linspace(-np.deg2rad(30), np.deg2rad(30), 2)
p = np.linspace(-1.2, 1.2, 2)
q = np.linspace(-1.2, 1.2, 2)
r = np.linspace(-0.3925, 0.3925, 2)
cases = list(itertools.product(alpha, beta, d_e, d_a, d_r, p, q, r))
# cases = list(itertools.product(alpha, beta))
def database(params):
    alpha = params[0]
    beta = params[1]
    d_e = params[2]
    d_a = params[3]
    d_r = params[4]
    p = params[5]
    q = params[6]
    r = params[7]
    rates = [p, q, r]
    my_scene.set_aircraft_state(state={"alpha" : alpha,
                                       "beta" : beta,
                                       "angular_rates" : rates,
                                       "velocity" : 222.5211})
    my_scene.set_aircraft_control_state(control_state={"elevator" : d_e,
                                                       "aileron" : d_a,
                                                       "rudder" : d_r})
    FM = my_scene.solve_forces()["F16"]["total"]
    CL = FM["CL"]
    CD = FM["CD"]
    CS = FM["CS"]
    Cl = FM["Cl"]
    Cm = FM["Cm"]
    Cn = FM["Cn"]
    print(CL)
    return [CL, CD, CS, Cl, Cm, Cn]

t0 = time.time()
for i in range(len(cases)):
    res = database(cases[i])
t1 = time.time()
print(t1 - t0)
# t0 = time.time()
# with multiprocessing.Pool() as pool:
#     results = pool.map(database, cases)
# t1 = time.time()
# print(t1 - t0)
