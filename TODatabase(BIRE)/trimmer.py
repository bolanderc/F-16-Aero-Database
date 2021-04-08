#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:28:04 2021

@author: christian
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator as rgi
import pandas as pd
import scipy.optimize as optimize
import matplotlib.pyplot as plt


class Trim:
    def __init__(self, V, rho):
        self.W = 20500.
        self.g = 32.2
        self.h_rot = np.zeros((3, 3))
        self.Ixx = 9496.  # slug-ft^2
        self.Iyy = 55814.  # slug-ft^2
        self.Izz = 63100.  # slug-ft^2
        self.Ixz = 982.  # slug-ft^2
        self.Ixy = 0.
        self.Iyz = 0.
        self.hx = 160.  # slug-ft^2/s
        self.h_rot[1, 2] = -self.hx
        self.h_rot[2, 1] = self.hx
        self.V = V
        self.rho = rho
        self.S_w = 300
        self.nd_coeff = 0.5*rho*V*V*self.S_w
        self.b_w = 30.
        self.c_w = 11.32

    def import_data(self, fn):
        aoa_lim = 15
        beta_lim = 15
        da_lim = 21.5/4.
        de_lim = 25
        dr_lim = 30
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
        if fn[-4:] == '.csv':
            self.data = pd.read_csv(fn, delimiter=',')
            self.data.sort_values(by=['AOA', 'Beta', 'd_e', 'd_a', 'd_r', 'p', 'q', 'r'], inplace=True)
            self.data.to_csv("./TODatabase_linear_sorted.csv")
            self.data_array = np.zeros((5, 5, 5, 5, 5, 5, 5, 5, 6))
            it = 0
            for i in range(num_pts):
                for j in range(num_pts):
                    for k in range(num_pts):
                        for m in range(num_pts):
                            for n in range(num_pts):
                                for pp in range(num_pts):
                                    for qq in range(num_pts):
                                        for rr in range(num_pts):
                                            self.data_array[i, j, k, m, n, pp, qq, rr, 0] = self.data.iat[it, 8]
                                            self.data_array[i, j, k, m, n, pp, qq, rr, 1] = self.data.iat[it, 9]
                                            self.data_array[i, j, k, m, n, pp, qq, rr, 2] = self.data.iat[it, 10]
                                            self.data_array[i, j, k, m, n, pp, qq, rr, 3] = self.data.iat[it, 11]
                                            self.data_array[i, j, k, m, n, pp, qq, rr, 4] = self.data.iat[it, 12]
                                            self.data_array[i, j, k, m, n, pp, qq, rr, 5] = self.data.iat[it, 13]
                                            it += 1
            np.save("./TODatabase_linear.npy", self.data_array)
        elif fn[-4:] == '.npy':
            self.data_array = np.load(fn)
            mask = np.isnan(self.data_array)
            self.data_array[mask] = np.interp(np.flatnonzero(mask),
                                              np.flatnonzero(~mask),
                                              self.data_array[~mask])
        self.CX_s = rgi((alpha, beta, d_e, d_a, d_r, p, q, r),
                        self.data_array[:, :, :, :, :, :, :, :, 0],
                        bounds_error=False, fill_value=None)
        self.CY_s = rgi((alpha, beta, d_e, d_a, d_r, p, q, r),
                        self.data_array[:, :, :, :, :, :, :, :, 1],
                        bounds_error=False, fill_value=None)
        self.CZ_s = rgi((alpha, beta, d_e, d_a, d_r, p, q, r),
                        self.data_array[:, :, :, :, :, :, :, :, 2],
                        bounds_error=False, fill_value=None)
        self.Cl_s = rgi((alpha, beta, d_e, d_a, d_r, p, q, r),
                        self.data_array[:, :, :, :, :, :, :, :, 3],
                        bounds_error=False, fill_value=None)
        self.Cm_s = rgi((alpha, beta, d_e, d_a, d_r, p, q, r),
                        self.data_array[:, :, :, :, :, :, :, :, 4],
                        bounds_error=False, fill_value=None)
        self.Cn_s = rgi((alpha, beta, d_e, d_a, d_r, p, q, r),
                        self.data_array[:, :, :, :, :, :, :, :, 5],
                        bounds_error=False, fill_value=None)

    def calc_fm_s(self, params):
        CX = self.CX_s(params)
        CY = self.CY_s(params)
        CZ = self.CZ_s(params)
        Cl = self.Cl_s(params)
        Cm = self.Cm_s(params)
        Cn = self.Cn_s(params)
        return [CX, CY, CZ, Cl, Cm, Cn]

    def eqs_of_motion(self, params, opt=True):
        [C_T, alpha, beta, d_e, d_a, d_r] = params
#        for i in range(len(params) - 1):
#            if abs(params[i + 1]) > self.param_lims[i]:
#                params[i + 1] = np.sign(params[i + 1])*self.param_lims[i]
        beta = 0.
        d_a = 0.
        d_r = 0.
        s_a = np.sin(np.deg2rad(alpha))
        c_a = np.cos(np.deg2rad(alpha))
        s_b = np.sin(np.deg2rad(beta))
        c_b = np.cos(np.deg2rad(beta))
        s_phi = np.sin(np.deg2rad(self.phi))
        c_phi = np.cos(np.deg2rad(self.phi))
        s_theta = np.sin(np.deg2rad(self.theta))
        c_theta = np.cos(np.deg2rad(self.theta))
        orientation = np.array([-s_theta, s_phi*c_theta, c_phi*c_theta])
        Rsb = np.zeros((3, 3))
        Rsb[0, 0] = c_a
        Rsb[0, 2] = -s_a
        Rsb[1, 1] = 1.
        Rsb[2, 0] = s_a
        Rsb[2, 2] = c_a
        u, v, w = self.V*np.array([c_a*c_b, s_b, s_a*c_b])
        p, q, r = (self.g*s_phi*c_theta)/(u*c_theta*c_phi + w*s_theta)*orientation
        if not opt:
            self.p_trim = p
            self.q_trim = q
            self.r_trim = r
        print('In: {:{w}.{p}f} {:{w}.{p}f} {:{w}.{p}f} {:{w}.{p}f} {:{w}.{p}f} {:{w}.{p}f} {:{w}.{p}f} {:{w}.{p}f}'.format(alpha, beta, d_e, d_a, d_r, p, q, r, w=13, p=6 ))
        Cfm_s = self.calc_fm_s([alpha, beta, d_e, d_a, d_r, p, q, r])
        print("Out:", *list(map("{: 13.6f} {: 13.6f} {: 13.6f} {: 13.6f} {: 13.6f} {: 13.6f}".format,*Cfm_s)))
        F_b = self.nd_coeff*np.matmul(Rsb, Cfm_s[:3]).T[0]
        F_b[0] += C_T*self.nd_coeff
        M_b = self.nd_coeff*np.matmul(Rsb, Cfm_s[3:]).T[0]
        M_b[0] *= self.b_w
        M_b[1] *= self.c_w
        M_b[2] *= self.b_w
        F_grav = self.W*orientation
        F_corr = (self.W/self.g)*np.array([r*v - q*w, p*w - r*u, q*u - p*v])
        F_tot = F_b + F_grav + F_corr
        M_rot = np.matmul(self.h_rot, [p, q, r])
        M_corr = np.array([(self.Iyy - self.Izz)*q*r + self.Iyz*(q*q - r*r) + self.Ixz*p*q - self.Ixy*p*r,
                           (self.Izz - self.Ixx)*p*r + self.Ixz*(r*r - p*p) + self.Ixy*q*r - self.Iyz*p*q,
                           (self.Ixx - self.Iyy)*p*q + self.Ixy*(p*p - q*q) + self.Iyz*p*r - self.Ixz*q*r])
        M_tot = M_b + M_rot + M_corr
        print("Params: {:>8.6f}\t{:>8.6f}\t{:>8.6f}\t{:>8.6f}\t{:>8.6f}\t{:>8.6f}".format(*params))
        print("FM: {:>8.6f}\t{:>8.6f}\t{:>8.6f}\t{:>8.6f}\t{:>8.6f}\t{:>8.6f}\n".format(*np.concatenate((F_tot, M_tot))))
        return np.concatenate((F_tot, M_tot))

    def general_trimmer(self, **kwargs):
        self.alpha = kwargs.get("alpha", None)
        self.beta = kwargs.get("beta", None)
        self.d_e = kwargs.get("d_e", None)
        self.d_a = kwargs.get("d_a", None)
        self.d_r = kwargs.get("d_r", None)
        self.p = kwargs.get("p", None)
        self.q = kwargs.get("q", None)
        self.r = kwargs.get("r", None)
        self.theta = kwargs.get("theta", None)
        self.phi = kwargs.get("phi", None)
        aoa_lim = 15
        beta_lim = 15
        da_lim = 21.5/4.
        de_lim = 25
        dr_lim = 30
        self.param_lims = [aoa_lim, beta_lim, de_lim, da_lim, dr_lim]
        state = [self.alpha, self.beta, self.d_e,
                 self.d_a, self.d_r, self.p,
                 self.q, self.r, self.theta, self.phi]
        bounds = [(None, None), (-aoa_lim, aoa_lim),
                  (-beta_lim, beta_lim), (-de_lim, de_lim),
                  (-da_lim, da_lim), (-dr_lim, dr_lim)]
        self.trim_solution = optimize.fsolve(self.eqs_of_motion, [17.5, 0., -8.65, 0., 0., 0.])
#        self.trim_solution = optimize.fsolve(self.eqs_of_motion, np.zeros(6))
        print(self.eqs_of_motion(self.trim_solution, opt=False))

    def cpa_point_generator(self):
        pnts = 10
        de_rng = np.linspace(-self.param_lims[2], self.param_lims[2], pnts*5)
        da_rng = np.linspace(-self.param_lims[3], self.param_lims[3], pnts)
        dr_rng = np.linspace(-self.param_lims[4], self.param_lims[4], pnts)
        self.Cl_cpa = np.zeros((pnts, pnts, pnts))
        self.Cm_cpa = np.zeros((pnts, pnts, pnts))
        self.Cn_cpa = np.zeros((pnts, pnts, pnts))
        alpha_t = self.trim_solution[1]
        beta_t = 0.
        de_t = self.trim_solution[3]
        da_t = 0.
        dr_t = 0.
        rates = [self.p_trim, self.q_trim, self.r_trim]
        c_a = np.cos(np.deg2rad(alpha_t))
        s_a = np.sin(np.deg2rad(alpha_t))
        for i in range(pnts):
            for j in range(pnts):
                for k in range(pnts):
                    params = [alpha_t, beta_t, de_rng[i], da_rng[j], dr_rng[k], *rates]
                    x = self.calc_fm_s(params)[3:]
                    self.Cl_cpa[i, j, k] = x[0]*c_a - x[2]*s_a
                    self.Cm_cpa[i, j, k] = x[1]
                    self.Cn_cpa[i, j, k] = x[2]*c_a + x[0]*s_a
        params[2] = de_t
        params[3] = da_t
        params[4] = dr_t
        x = self.calc_fm_s(params)[3:]
        self.Cl_t = x[0]*c_a - x[2]*s_a
        self.Cm_t = x[1]
        self.Cn_t = x[2]*c_a + x[0]*s_a
        self.pt_cloud = np.stack((np.append(self.Cl_cpa.flatten(), [self.Cl_t]), np.append(self.Cm_cpa.flatten(), [self.Cm_t]), np.append(self.Cn_cpa.flatten(), [self.Cn_t])), axis=-1)
        np.save("pt_cloud.npy", self.pt_cloud)




theta = 0.
phi = 0.
rho = 0.0023084
V = 222.5211
case = Trim(V, rho)
case.import_data("./TODatabase_linear.npy")
case.general_trimmer(theta=theta, phi=phi)
case.cpa_point_generator()
