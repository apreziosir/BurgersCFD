#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions for Burgers 2D equation
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
"""

import numpy as np

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
    
    # Initial conditions for Burgers 2D test equation
    u0 = -np.sin(np.pi * X)
    v0 = -np.sin(np.pi * Y)
    
    return [u0, v0]