#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:40:33 2021

@author: christian
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

plt.close('all')

def point_cloud_plot(fn):
    pts = np.load(fn)
    env_fig = plt.figure()
    env_ax = env_fig.add_subplot(111, projection='3d')
    env_ax.scatter(pts[:-2, 0], pts[:-2, 1], pts[:-2, 2])
    env_ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color='r', s=40)
    env_ax.set_xlabel("Rolling Moment")
    env_ax.set_ylabel("Pitching Moment")
    env_ax.set_zlabel("Yawing Moment")
    plt.show()

    lat_fig = plt.figure()
    lat_ax = lat_fig.add_subplot(111)
    lat_ax.scatter(pts[:-2, 0], pts[:-2, 2])
    lat_ax.scatter(pts[-1, 0], pts[-1, 2], color='r', s=20)
    lat_ax.set_xlabel("Rolling Moment")
    lat_ax.set_ylabel("Yawing Moment")
    plt.show()

    lon_fig = plt.figure()
    lon_ax = lon_fig.add_subplot(111)
    lon_ax.scatter(pts[:-2, 0], pts[:-2, 1])
    lon_ax.scatter(pts[-1, 0], pts[-1, 1], color='r', s=20)
    lon_ax.set_xlabel("Rolling Moment")
    lon_ax.set_ylabel("Pitching Moment")
    plt.show()

    last_fig = plt.figure()
    last_ax = last_fig.add_subplot(111)
    last_ax.scatter(pts[:-2, 2], pts[:-2, 1])
    last_ax.scatter(pts[-1, 2], pts[-1, 1], color='r', s=20)
    last_ax.set_xlabel("Yawing Moment")
    last_ax.set_ylabel("Pitching Moment")
    plt.show()

def surface_plot(fn):
    pts = np.load(fn)
    hull = ConvexHull(pts)
    fig_sf = plt.figure()
    ax_sf = fig_sf.add_subplot(111, projection="3d")
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax_sf.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")



fn = './pt_cloud.npy'
point_cloud_plot(fn)
surface_plot(fn)
