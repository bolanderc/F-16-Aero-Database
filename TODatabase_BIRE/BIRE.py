#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:48:40 2021

@author: christian
"""

import numpy as np
import json
import machupX as mx


def create_inputs(inp_dir, d_r):
    rotation_angle = str(int(d_r))

    f_inp = open(inp_dir + 'BIRE_input.json',)
    inp_data = json.load(f_inp)

    f_air = open(inp_dir + 'BIRE_airplane.json',)
    air_data = json.load(f_air)

    bire_dihedral = [[0.0, 0.0],
                     [0.26818, 0.0],
                     [0.26818, 0.0],
                     [0.92617, 0.0],
                     [0.92617, 0.0],
                     [1.0, 0.0]]
    bire_left = [row[:] for row in bire_dihedral]
    for row in range(len(bire_dihedral)):
        bire_left[row][1] = d_r
    bire_right = [row[:] for row in bire_dihedral]
    for row in range(len(bire_right)):
        bire_right[row][1] = -d_r
    bire_top = 90. + d_r
    bire_bottom = -d_r - 90
    air_data["wings"]["BIRE_left"]["dihedral"] = bire_left
    air_data["wings"]["BIRE_right"]["dihedral"] = bire_right
    air_data["wings"]["BIRE_top"]["dihedral"] = bire_top
    air_data["wings"]["BIRE_bottom"]["dihedral"] = bire_bottom

    new_air_fn = inp_dir + 'BIRE_airplane_dr_' + rotation_angle + '.json'
    with open(new_air_fn, 'w') as fp:
        json.dump(air_data, fp)

    inp_data["scene"]["aircraft"]["BIRE"]["file"] = new_air_fn
    new_inp_fn = inp_dir + 'BIRE_input_dr_' + rotation_angle + '.json'
    with open(new_inp_fn, 'w') as fp:
        json.dump(inp_data, fp)


def create_slurms(slurm_dir, d_r, cluster_name='kp'):
    rotation_angle = str(int(d_r))
    slurm_exists = False
    try:
        f = open(slurm_dir + 'BIRE_database_dr' + rotation_angle + '.slurm',)
        slurm_exists = True
    except FileNotFoundError:
        f = open(slurm_dir + 'BIRE_database_dr' + rotation_angle + '.slurm', 'w')
    if slurm_exists:
        return
    else:
        header = ['#!/bin/bash\n',
                  '#SBATCH --time=5-00:00:00\n',
                  '#SBATCH --nodes=1\n',
                  '#SBATCH --account=usumae-' + cluster_name + '\n',
                  '#SBATCH --partition=usumae-' + cluster_name + '\n',
                  '#SBATCH -o TO_BIRE_' + rotation_angle + '-%j\n',
                  '#SBATCH --mail-user=christian.bolander@aggiemail.usu.edu\n',
                  '#SBATCH --mail-type=END\n\n',
                  'module purge\n',
                  'module load python/3.7.3\n\n']
        directory = 'cd /uufs/chpc.utah.edu/common/home/u0764388/BIRE/F-16-Aero-Database/TODatabase_BIRE\n'
        run = 'python BIREMUX.py > output-$SLURM_JOB_ID-TO_B.out'
        f.writelines(header)
        f.write(directory)
        f.write(run)

        f.close()


def bire_case(params, inp_dir):
    [alpha, beta, d_e, d_a, d_r, p, q, r] = params
    rotation_angle = str(int(d_r))
    try:
        f = open(inp_dir + 'BIRE_input_dr_' + rotation_angle + '.json',)
    except FileNotFoundError:
        create_inputs(inp_dir, d_r)
    input_file = inp_dir + 'BIRE_input_dr_' + rotation_angle + '.json'
    BIRE_scene = mx.Scene(input_file)
    forces_options = {'body_frame': True,
                      'stab_frame': False,
                      'wind_frame': True,
                      'dimensional': False,
                      'verbose': False}
    rates = [p, q, r]
    BIRE_scene.set_aircraft_state(state={"alpha": alpha,
                                         "beta": beta,
                                         "angular_rates": rates,
                                         "velocity": 222.5211})
    BIRE_scene.set_aircraft_control_state(control_state={"elevator": d_e,
                                                         "aileron": d_a})
    x = BIRE_scene.solve_forces(**forces_options)["BIRE"]["total"]
    fm = [x['CD'], x['CS'], x['CL'], x['Cl'], x['Cm'], x['Cn']]
    return fm

alpha = 10.
beta = 20.
d_e = 10.
d_a = 0.
d_r = 60.
p = 0.
q = 0.
r = 0.
params = [alpha, beta, d_e, d_a, d_r, p, q, r]
inp_dir = './'

FM = bire_case(params, inp_dir)
create_inputs(inp_dir, d_r)
create_slurms(inp_dir, d_r)