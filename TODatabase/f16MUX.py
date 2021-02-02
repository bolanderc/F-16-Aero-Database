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

# Input the json module to make displaying dictionaries more friendly
import json
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)

t0 = time.time()
# Define the input file. The input file will contain the path to the aircraft
# file, and so this does not need to be defined here.
input_file = "F16_input.json"

# Initialize Scene object. This contains the airplane and all necessary
# atmospheric data.
my_scene = MX.Scene(input_file)

# We are now ready to perform analyses. display_wireframe will let us look at 
# the aircraft we have created. To make sure we know where each lifting surface 
# is, we'll set show_legend to true.
my_scene.display_wireframe()
alpha = np.linspace(np.deg2rad(-10), np.deg2rad(10), 10)
CL = np.zeros(len(alpha))
for i in range(len(alpha)):
    my_scene.set_aircraft_state(state={"alpha" : alpha[i], "velocity" : 634.4133})
    FM_results = my_scene.solve_forces(dimensional=True, non_dimensional=True, verbose=True, report_by_segment=True)
    CL[i] = json.dumps(FM_results["F16"]["total"]["CL"])
t1 = time.time()
print(t1 - t0)
# trim_state = my_scene.pitch_trim(set_trim_state=True, verbose=True)
# print(json.dumps(trim_state["F16"], indent=4))
# derivs = my_scene.derivatives()
# print(json.dumps(derivs["F16"]))

# my_scene.distributions(make_plots=["section_CL"], show_plots=True)
# input_file = "F16htail_input.json"
# tail_scene = MX.Scene(input_file)
# tail_scene.display_wireframe()

# alpha = np.linspace(-np.deg2rad(10.), np.deg2rad(10.), 20)
# CL_full = np.zeros_like(alpha)
# CL_tail = np.zeros_like(alpha)

# Let's see what forces are acting on the airplane. We'll output just the total
# dimensional forces and moments acting on the airplane. Note we need to know 
# the name of the airplane to be able to access its data.
# for i in range(20):
#     my_scene.set_aircraft_state(state={"alpha" : alpha[i], "velocity" : 596.9097})
#     FM_results = my_scene.solve_forces(dimensional=True, non_dimensional=True, verbose=True, report_by_segment=True)
#     CL_full[i] = json.dumps(FM_results["F16"]["inviscid"]["FL"]["h_stab_left"], indent=4)
#     tail_scene.set_aircraft_state(state={"alpha" : alpha[i], "velocity" : 596.9097})
#     FM_results = tail_scene.solve_forces(dimensional=True, non_dimensional=True, verbose=True, report_by_segment=True)
#     CL_tail[i] = json.dumps(FM_results["F16"]["inviscid"]["FL"]["h_stab_left"], indent=4)
# downwash = CL_full/CL_tail
# a = np.rad2deg(alpha)
# plt.plot(a, downwash)

# dist_results = my_scene.distributions(make_plots=["section_CL"])

# Now let's get the airplane to its trim state in pitch. MachUpX will default to 
# Using the 'elevator' control to trim out the airplane. We can use set_trim_state 
# to have MachUpX set the trim state to be the new state of the airplane.
# trim_state = my_scene.pitch_trim(set_trim_state=True, verbose=True)
# print(json.dumps(trim_state["F16"], indent=4))
# derivs = my_scene.derivatives()
# print(json.dumps(derivs["F16"]))

# Now that we're trimmed, let's see what our aerodynamic derivatives are.
# alpha = np.linspace(-np.deg2rad(40.), np.deg2rad(40.), 20)
# SM = np.zeros_like(alpha)
# for i in range(20):
#     my_scene.set_aircraft_state(state={"alpha" : alpha[i], "velocity" : 596.9097})
#     derivs = my_scene.derivatives()
#     # print(json.dumps(derivs["F16"], indent=4))
#     SM[i] = -derivs["F16"]["stability"]["Cm,a"]/derivs["F16"]["stability"]["CL,a"]
# plt.plot(alpha, SM)
