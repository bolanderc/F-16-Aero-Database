#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Christian Bolander
MAE 6510
Standard Atmosphere Calculator
"""

import numpy as np


def atm_print():
    headerlines_si = (' Geometric Geopotential               ' +
                      '                           Speed of' +
                      '\n Altitude    Altitude   Temperature   ' +
                      'Pressure      Density       Sound\n' +
                      '   (m)         (m)          (K)       ' +
                      '(N/m**2)     (kg/m**3)      (m/s)' +
                      '\n--------------------------------------' +
                      '------------------------------------')
    print(headerlines_si)
    for i in range(51):
        h = float(i)*2000.
        z, t, p, d = statsi(h)
        a = np.sqrt(1.4*287.0528*t)
        print(int(h), int(z), t, p, d, a)
        headerlines_ee = (' Geometric Geopotential               ' +
                          '                           Speed of' +
                          '\n Altitude    Altitude   Temperature   ' +
                          'Pressure      Density       Sound\n' +
                          '   (ft)        (ft)         (R)     ' +
                          '(lbf/ft**2)  (slugs/ft**3)    (ft/s)' +
                          '\n--------------------------------------' +
                          '------------------------------------')
    print(headerlines_ee)
    for i in range(51):
        h = float(i)*5000.
        z, t, p, d = statee(h)
        tsi = t/1.8
        asi = np.sqrt(1.4*287.0528*tsi)
        a = asi/0.3048
        print(int(h), int(z), t, p, d, a)


def statsi(h):
    Psa = np.zeros(9)
    zsa = [0., 11000., 20000., 32000., 47000., 52000., 61000., 79000., 9.9e20]
    Tsa = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 252.65, 180.65,
           180.65]
    g0 = 9.80665
    R = 287.0528
    Re = 6356766.
    Psa[0] = 101325.
    z = Re*h/(Re+h)
    for i in range(1, 9):
        Lt = -(Tsa[i] - Tsa[i-1])/(zsa[i] - zsa[i-1])
        if Lt == 0:
            if z <= zsa[i]:
                t = Tsa[i-1]
                p = Psa[i-1]*np.exp(-g0*(z-zsa[i-1])/R/Tsa[i-1])
                d = p/R/t
                return z, t, p, d
            else:
                Psa[i] = Psa[i-1]*np.exp(-g0*(zsa[i] - zsa[i-1])/R/Tsa[i-1])
        else:
            ex = g0/R/Lt
            if z < zsa[i]:
                t = Tsa[i-1] - Lt*(z-zsa[i-1])
                p = Psa[i-1]*(t/Tsa[i-1])**ex
                d = p/R/t
                return z, t, p, d
            else:
                Psa[i] = Psa[i-1]*(Tsa[i]/Tsa[i-1])**ex
    t = Tsa[8]
    return z, t, p, d


def statee(h):
    hsi = h*0.3048
    zsi, tsi, psi, dsi = statsi(hsi)
    z = zsi/0.3048
    t = tsi*1.8
    p = psi*0.02088543
    d = dsi*0.001940320
    return z, t, p, d

