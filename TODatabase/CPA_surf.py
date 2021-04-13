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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from descartes import PolygonPatch
import alphashape
from scipy.spatial import Delaunay
from collections import defaultdict

plt.close('all')

def alpha_shape_3D(pos, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """

    tetra = Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra.vertices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))

    # Find tetrahedrals
    tetras = tetra.vertices[r<alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:,TriComb].reshape(-1,3)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
    Edges=Triangles[:,EdgeComb].reshape(-1,2)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)
    return Vertices,Edges,Triangles

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
    Cl = pts[:-1, 0]
    Cm = pts[:-1, 1]
    Cn = pts[:-1, 2]
    ClCn = []
    ClCm = []
    CnCm = []
    for i in range(len(Cl)):
        ClCn.append((Cl[i], -Cn[i]))
        ClCm.append((Cl[i], Cm[i]))
        CnCm.append((-Cn[i], Cm[i]))

    alpha_ClCn = alphashape.alphashape(ClCn, 5.0)
    fig_ClCn, ax_ClCn = plt.subplots()
    ax_ClCn.scatter(*zip(*ClCn))
    ax_ClCn.scatter(pts[-1, 0], pts[-1, 2], color='r')
    ax_ClCn.add_patch(PolygonPatch(alpha_ClCn, alpha=0.2, color='g'))

    alpha_ClCm = alphashape.alphashape(ClCm, 2.0)
    fig_ClCm, ax_ClCm = plt.subplots()
    ax_ClCm.scatter(*zip(*ClCm))
    ax_ClCm.scatter(pts[-1, 0], pts[-1, 1], color='r')
    ax_ClCm.add_patch(PolygonPatch(alpha_ClCm, alpha=0.2, color='g'))

    alpha_CnCm = alphashape.alphashape(CnCm, 5.0)
    fig_CnCm, ax_CnCm = plt.subplots()
    ax_CnCm.scatter(*zip(*CnCm))
    ax_CnCm.scatter(pts[-1, 2], pts[-1, 1], color='r')
    ax_CnCm.add_patch(PolygonPatch(alpha_CnCm, alpha=0.2, color='g'))

    verts, edges, tris = alpha_shape_3D(pts, 2.0)
    fig = plt.figure ()
    ax = fig.add_subplot (1, 1, 1, projection = '3d')

    ax.add_collection3d (Poly3DCollection (pts[verts]))



fn = './pt_cloud.npy'
#point_cloud_plot(fn)
surface_plot(fn)
