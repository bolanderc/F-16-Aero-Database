import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import machupX as mx
import matplotlib.pyplot as plt
import numpy as np
import stdatmos as atmos
import scipy.optimize as optimize
import itertools
from multiprocessing import Pool
import ZachsModules as zm


def database(params):
    alpha = params[0]
    beta = params[1]
    d_e = params[2]
    d_a = params[3]
    p = params[4]
    q = params[5]
    r = params[6]
    rates = [p, q, r]
    my_scene.set_aircraft_state(state={"alpha": alpha,
                                       "beta": beta,
                                       "angular_rates": rates,
                                       "velocity": 222.5211})
    my_scene.set_aircraft_control_state(control_state={"elevator": d_e,
                                                       "aileron": d_a})
    try:
        x = my_scene.solve_forces(**forces_options)["F16"]["total"]
        fm = [x['Cx_s'], x['Cy_s'], x['Cz_s'], x['Cl_s'], x['Cm_s'], x['Cn_s']]
    except mx.exceptions.SolverNotConvergedError:
        fm = [None] * 6
    return (*params, *fm)


input_file = "BIRE_input.json"
my_scene = mx.Scene(input_file)
forces_options = {'body_frame': False,
                  'stab_frame': True,
                  'wind_frame': False,
                  'dimensional': False,
                  'verbose': False}
aoa_lim = 15
beta_lim = 15
da_lim = 5.375
de_lim = 25
dr_lim = 90
pq_lim = 1.2
r_lim = 0.3925
num_pts = 5
alpha = np.linspace(-aoa_lim, aoa_lim, num_pts)
beta = np.linspace(-beta_lim, beta_lim, num_pts)
d_e = np.linspace(-de_lim, de_lim, num_pts)
d_a = np.linspace(-da_lim, da_lim, num_pts)
d_r = np.linspace(-dr_lim, dr_lim, num_pts)
p = np.linspace(-pq_lim, pq_lim, num_pts)
q = np.linspace(-pq_lim, pq_lim, num_pts)
r = np.linspace(-r_lim, r_lim, num_pts)
cases = list(itertools.product(alpha, beta, d_e, d_a, p, q, r))

if __name__ == '__main__':

    fn = 'TODatabase.csv'
    f = open(fn, 'w')
    f.write(zm.io.csvLineWrite('AOA',
                               'Beta',
                               'd_e',
                               'd_a',
                               'd_r',
                               'p',
                               'q',
                               'r',
                               'Cx_s',
                               'Cy_s',
                               'Cz_s',
                               'Cl_s',
                               'Cm_s',
                               'Cn_s'))
    f.close()

    bat = 5000
    chu = 2

    zm.nm.runCases(database, cases, fn, nBatch=bat, chunkSize=chu,
                   progKW={'title': 'Running Cases: {}/batch, {}/chunck'.format(bat, chu)})
