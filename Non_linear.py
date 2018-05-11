#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Handlers for the non linear operator in the Burgers' equation. Three forms of 
the operator implemented: 
1. Primitive variables
2. Divergence form
3. Skew-symmetric
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
Mayo de 2018
"""

import numpy as np

# DIIFERENTIATING IN THE X DIRECTION - UPWIND OR HIGH ORDER UPWIND
def diff_x(u, dx):
    
# ==============================================================================
# PARTS OF THE FUNCTION THAT CAN BE CHANGED IN ORDER TO MODIFY THE SCHEME
# ==============================================================================
    
    # Upwind method. If u_m = 0, it is a normal upwind scheme
    # If u_m = 1, it is a precision upwind from fletcher, 
    # If u_m = 2, it is a three point upwind scheme from Chapra
    # If u_m = 3, it is a 5 point centered scheme from Chapra also
    u_m = 3
    
    # Modifiable part of the routine with a coefficient q given in Fletcher 1991
    # When q = 0.5 the scheme becomes a third order scheme (only applies when 
    # the high order method from Fletcher is implemented)
    q = 0.5
# ==============================================================================
    
    # Defining vector that stores the values of the differentiation
    n = len(u)
    diffx = np.zeros(n)
    
    # Differentiating the first and last elements of the velocity vector. There
    # is no way to make upwinding here (same for normal upwind and more precise
    # upwind)
    diffx[0] = (u[1] - u[0]) / dx
    diffx[n - 1] = (u[n - 1] - u[n - 2]) / dx
    
    # Differentiating the second and second to last elements of the vector (also 
    # this is the same case for the three cases of upwinding)
    if u[1] >= 0:
        
        diffx[1] = (u[1] - u[0]) / dx 
        
    else:
        
        diffx[1] = (u[2] - u[1]) / dx
        
    if u[n - 2] >= 0:
        
        diffx[n - 2] = (u[n - 2] - u[n - 3]) / dx
        
    else: 
        
        diffx[n - 2] = (u[n - 1] - u[n - 2]) / dx
    
    # This part applies the high order upwind scheme for the rest of the 
    # nodes of the problem
    if u_m == 1:
        
        # Looping over the elements of the inside part of the vector (the 
        # outside is managed in a different way since it is a high order scheme)
        for i in range(2, n - 2):
            
            # Selecting the upwind implementation (from left to right)
            if u[i] >= 0:
            
                diffx[i] = (u[i + 1] - u[i - 1]) / (2 * dx) 
                diffx[i] += q * (u[i - 2] - 3 * u[i - 1] + 3 * u[i] - u[i + 1])\
                / (3 * dx)
                
            # Upwind when velocity is negative (from right to left)
            else:
                
                diffx[i] = (u[i + 1] - u[i - 1]) / (2 * dx)
                diffx[i] += q * (u[i - 1] - 3 * u[i] + 3 * u[i + 1] - u[i + 2])\
                / (3 * dx)
        
    # This part implements a two point upwind scheme O(h)
    elif u_m == 0: 
            
        for i in range(2, n - 2):
                
            # upwind for positive x velocity
            if u[i] >= 0:
                    
                diffx[i] = (u[i] - u[i - 1]) / dx
                
            # Upwind for negative x velocity
            else:
                    
                diffx[i] = (u[i + 1] - u[i]) / dx
                    
    # This part implements a three point upwind scheme where it is possible
    elif u_m == 2:
        
        for i in range(2, n - 2):
            
            # Upwind for positive x velocity
            if u[i] >= 0:
                
                diffx[i] = (3 * u[i] - 4 * u[i - 1] + u[i - 2]) / (2 * dx)
                
            # Upwind for negative x velocity
            else: 
                
                diffx[i] = (-u[i + 2] + 4 * u[i + 1] - 3 * u[i]) / (2 * dx)
    else: 
        
        for i in range(2, n - 2):
            
            diffx[i] = (-u[i + 2] + 8 * u[i + 1] - 8 * u[i - 1] + u[i - 2]) / \
            (12 * dx)
                        
    return diffx

# ==============================================================================
# Solving the non linear term using primitive variables. u (dot) nabla u
# ==============================================================================

def prim_var(u0, dx, dt):
    
    u1 = np.multiply(u0, 1 - dt * diff_x(u0, dx))
        
    return u1

# ==============================================================================
# Solving the non linear term using the divergence form
# ==============================================================================

def div_f(u0, dx, dt):
    
    ut = np.multiply(u0, u0)
    u1 = u0 - dt * 0.5 * diff_x(ut, dx)
    
    return u1

# ==============================================================================
# Skew symmetric form
# ==============================================================================

def Skew_sym(u0, dx, dt):
    
    ut1 = np.multiply(u0, diff_x(u0, dx))
    ut2 = np.multiply(u0, u0)
    ut3 = 0.5 * diff_x(ut2, dx)
    
    u1 = u0 - dt * 0.5 * (ut1 + ut3)
    
    return u1