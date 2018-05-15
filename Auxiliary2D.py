#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions for Burgers 2D equation
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
"""

import numpy as np
import scipy.sparse as sp

# ==============================================================================
# ROUTINES TO IDENTIFY THE NODES OF THE DOMAIN THAT WILL BE DE PHYSICAL 
# BOUNDARY AND DE MOST EXTERNAL RING OF NODES CLOSE TO THE BOUNDARY 
# ==============================================================================

# Top boundary & ring
def Top_BR(Nx, Ny):
    
    # Top boundary
    T_B = np.linspace((Ny - 1) * Nx, Nx * Ny - 1, Nx)
    
    T_R = T_B - Nx                          # Bottom ring
    T_R = np.delete(T_R, 0)                 # Taking out first
    T_R = np.delete(T_R, -1)                # Taking out last  
    
    return [T_B, T_R]

# Bottom boundary & ring
def Bot_BR(Nx, Ny):
    
    # Bottom boundary
    B_B = np.linspace(0, Nx - 1, Nx)
    
    # Bottom ring
    B_R = B_B + Nx                          # Bottom ring
    B_R = np.delete(B_R, 0)                 # Taking out first
    B_R = np.delete(B_R, -1)                # Taking out last
    
    return [B_B, B_R]

# Left boundary
def Left_BR(Nx, Ny):
    
    # Left boundary
    L_B = np.linspace(Nx, (Ny - 2) * Nx, Ny - 2)                
    
    L_R = L_B + 1                           # Left ring
    L_R = np.delete(L_R, 0)                 # Taking out first
    L_R = np.delete(L_R, -1)                # Taking out last
    
    return [L_B, L_R]
    
# Right boundary
def Right_BR(Nx, Ny):
    
    # Right boundary
    L_B = np.linspace(Nx, (Ny - 2) * Nx, Ny - 2)
    R_B = L_B + Nx
    del(L_B)
    
    R_R = R_B - 1                           # Right ring
    R_R = np.delete(R_R, 0)                 # Taking out first
    R_R = np.delete(R_R, -1)                # Taking out last
   
    return [R_B, R_R]

# ==============================================================================
# IMPOSING INITIAL CONDITION
# ==============================================================================

def I_C(X, Y):
    
    # Initial conditions for Burgers 2D test equation - use just one and set the 
    # other one to 0 for testing purposes
    u0 = -np.sin(np.pi * X)
#    v0 = -np.sin(np.pi * Y)

    # Zero condition
#    u0 = 0 * X * Y
    v0 = 0 * X * Y
    
    return [u0, v0]

# ==============================================================================
# ASSEMBLYING MATRICES FOR VISCOUS TERM IN BOTH DIRECTIONS (assuming boundary
# conditions given in the arrays of BC)
# ==============================================================================
    
def Ass_matrix(K, Nx, Ny, Sx, Sy, BC0):
    
    # Defining matrix according to number of nodes in the domain
    nn = Nx * Ny
    
    # Looking for boundary nodes in the domain according to Nx and Ny
    [B_B, B_R] = Bot_BR(Nx, Ny)
    [T_B, T_R] = Top_BR(Nx, Ny)
    [L_B, L_R] = Left_BR(Nx, Ny)
    [R_B, R_R] = Right_BR(Nx, Ny)
   
    # Filling matrix with typical nodal values (then the boundary nodes are 
    # replaced with rows equal to 0 and changed)
    for i in range(int(B_R[0]), int(T_R[-1])):
        
        K[i, i] = 1 + 2 * Sx + 2 * Sy
        K[i, i + 1] = -Sx
        K[i, i - 1] = -Sx
        K[i, i + Nx] = -Sy
        K[i, i - Nx] = -Sy
        
    # Setting the value of the rows that represent boundary elements to 0
    K[B_B, :] = np.zeros(nn)
    K[T_B, :] = np.zeros(nn)
    K[R_B, :] = np.zeros(nn)
    K[L_B, :] = np.zeros(nn)
    
    # Looking ofr each case of boundary condition and setting it to the matrix
    # in each one of the boundary nodes
    
    # Bottom
    if BC0[0] == 0 : K[B_B, B_B] = 1
        
    elif BC0[0] == 1:
        
        K[B_B, B_B] = -1
        K[B_B, B_B + Nx] = 1
    
    # Top
    if BC0[1] == 0 : K[T_B, T_B] = 1
        
    elif BC0[1] == 1:
        
        K[T_B, T_B] = 1
        K[T_B, T_B - Nx] = -1
        
    # Left
    if BC0[2] == 0 : K[L_B, L_B] = 1
        
    elif BC0[2] == 1:
        
        K[L_B, L_B] = -1
        K[L_B, L_B + 1] = 1
    
    # Right
    if BC0[3] == 0 : K[R_B, R_B] = 1
    
    elif BC0[3] == 1:
        
        K[R_B, R_B] = 1
        K[R_B, R_B - 1] = -1
    
    return K