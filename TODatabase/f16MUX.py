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

t0 = time.time()
input_file = "F16_input.json"
my_scene = Scene(input_file)
print("Instance", time.time() - t0)
#my_scene.display_wireframe(show_vortices=True)
#my_scene.export_stl(filename="F16.stl")
alpha = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
beta = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
#d_e = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
#d_a = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
#d_r = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
#p = np.linspace(-1.57, 1.57, 5)
#q = np.linspace(-1.57, 1.57, 5)
#r = np.linspace(-0.3925, 0.3925, 5)
#cases = list(itertools.product(alpha, beta, d_e, d_a,))
cases = list(itertools.product(alpha, beta))
def database(params):
    my_scene.set_aircraft_state(state={"alpha" : params,
                                       "velocity" : 222.5211})
#        my_scene.set_aircraft_control_state(control_state={"elevator" : d_e})
    FM = my_scene.solve_forces()["F16"]["total"]
    return FM

#t0 = time.time()
#for i in range(len(alpha)):
#    database(alpha[i])
#t1 = time.time()
#print(t1 - t0)
t0 = time.time()
with Pool(processes=8) as pool:
    coeffs = pool.map(database, alpha)
t1 = time.time()
print(t1 - t0)
