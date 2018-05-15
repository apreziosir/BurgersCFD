#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function for numerical integration of the Analytical solution for the Burgers'
1D equation (Cole-Hopf transformation).
The integration is performed with Gaussian quadrature, estimating the 
GAuss-Lobatto integration points and weights. 
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
May 2018
"""

import numpy as np

def Analyt(xn, yn, t, nu):
    
    # Precision parameters (tweakeable part of the code for precision purposes).
    # The larger the number of points the more precise the integration
    # However, remember that the integrand is infinite and a trascendental 
    # function
    err = 85                             # Points for integration
    inf = -1                             # Inferior limit for integral
    sup = 1                              # Superior limit for integral
    
    # Generating vector of points and weights
    [y, W] = np.polynomial.legendre.leggauss(err)
    [X, Y] = np.meshgrid(xn, yn)
    
    # Defining vector that will sotre the anaytical values for velocity 
    U = 0 * X * Y
    ua = np.zeros(len(xn))
    
    # Calculating coefficients for change of variable according to integration
    # limits (new variable named y). Also calculating d_eta
    yk = (inf + sup) / 2                # Constant part (usually 0, see limits)
    yc = (sup - inf) / 2                # Coefficient that goes with y
    d_eta = yc                          # dx = yc * d_eta
    
    # Looping over x array to get velocity in every x position
    for o in range(0, len(y)):
        
        for i in range(0, len(xn)):
            
            # Starting values for calculating integrals when iterating in j loop
            Int1 = 0
            Int2 = 0
            
            # Looping over the points of the function (integration points)
            for j in range(0, err):
                
                # Calculating common values for points 
                y1 = yk + yc * y[j]
                
                x_y = xn[i] - y1
                
                # Calculating the factors of the numeratorintegral
                temp1 = np.sin(np.pi * x_y)
                
                temp2 = np.exp(np.cos(np.pi * x_y) / (-2 * np.pi * nu))
                
                temp3 = np.exp(y1 ** 2 / (-4 * nu * t)) * d_eta
                
                Int1 += (temp1 * temp2 * temp3 * W[j])
                
                # Calculating the factors for the denominator integral
                Int2 += (temp2 * temp3 * d_eta * W[j])
                
            # Estimating value of the velocity in the point analyzed
            ua[i] = -Int1 / Int2
            
    # Start to fill the desired velocity field
    for i in range(0, len(yn)):
        
        U[:, i] = ua
       
    return U