#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:14:47 2021

@author: christian
"""

import airfoil_db as adb
import matplotlib.pyplot as plt
import numpy as np
import math as m
from mpl_toolkits.mplot3d import Axes3D

if __name__=="__main__":

    # Where the airfoil geometry is stored
    geometry_file = "./5biconvex.txt"

    # Declare input structure
    airfoil_input = {
        "type" : "database",
        "geometry" : {
            "NACA" : "0005"
        },
        "trailing_flap_type" : "linear"
    }

    # Initialize airfoil
    airfoil = adb.Airfoil("5biconvex", airfoil_input, verbose=True)
    # Setting verbose to true will let us see how well the camber line/outline are detected and fit.
    # We recommend turning this on for new airfoils

    # Set up database degrees of freedom
    # With Rey computed from MAC of 5.906 ft
    dofs = {
        "alpha" : {
            "range" : [m.radians(-5.0), m.radians(5.0)],
            "steps" : 21,
            "index" : 1
        },
        "Rey" : 8160700.0,
        "Mach" : 0.2
    }

    # Generate database using xfoil
    airfoil.generate_database(degrees_of_freedom=dofs, N=300, max_iter=300, show_xfoil_output=False)
    airfoil.export_database(filename="horizontal_tail_database.txt")

    # If you just want a data database, stop here.
    # If you'd like to create a set of polynomial fits to your database, this next set of code will do that

    # Declare the order of the fit for each degree of freedom for each coefficient
    CL_fit_orders = {
        "alpha" : 1
    }

    CD_fit_orders = {
        "alpha" : 2
    }

    Cm_fit_orders = {
        "alpha" : 1
    }
    airfoil.generate_linear_model(plot_model=True, Rey=8160700.0, Mach=0.2)

    # Generate fits
    airfoil.generate_polynomial_fit(CL_degrees=CL_fit_orders, CD_degrees=CD_fit_orders, Cm_degrees=Cm_fit_orders)

    # Alternatively, you can have it automatically detect the order of fit that would be best for each
    #airfoil.generate_polynomial_fit(CL_degrees="auto", CD_degrees="auto", Cm_degrees="auto", max_order=6)

    # Export fits
    airfoil.export_polynomial_fits(filename="horizontal_tail_fits.json")

    # To use these database files in MachUpX, you'll want to declare the airfoil as follows
    #   "my_airfoil" : {
    #       "type" : "database",
    #       "input_file" : "my_airfoil_database.txt",
    #       "geometry" : {
    #           "outline_points" : "my_airfoil_geom.txt"
    #       }
    #   }
    #
    # or
    #
    #   "my_airfoil" : {
    #       "type" : "poly_fit",
    #       "input_file" : "my_airfoil_fits.json",
    #       "geometry" : {
    #           "outline_points" : "my_airfoil_geom.txt"
    #       }
    #   }
