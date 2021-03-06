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
    T_B = np.transpose(T_B)
    T_B = T_B.astype(int)
    
    T_R = T_B - Nx                          # Bottom ring
    T_R = np.delete(T_R, 0)                 # Taking out first
    T_R = np.delete(T_R, -1)                # Taking out last  
    T_R = T_R.astype(int)
    
    return [T_B, T_R]

# Bottom boundary & ring
def Bot_BR(Nx, Ny):
    
    # Bottom boundary
    B_B = np.linspace(0, Nx - 1, Nx)
    B_B = np.transpose(B_B)
    B_B = B_B.astype(int)
    
    # Bottom ring
    B_R = B_B + Nx                          # Bottom ring
    B_R = np.delete(B_R, 0)                 # Taking out first
    B_R = np.delete(B_R, -1)                # Taking out last
    B_R = B_R.astype(int)
    
    return [B_B, B_R]

# Left boundary
def Left_BR(Nx, Ny):
    
    # Left boundary
    L_B = np.linspace(Nx, (Ny - 2) * Nx, Ny - 2)                
    L_B = np.transpose(L_B)
    L_B = L_B.astype(int)
    
    L_R = L_B + 1                           # Left ring
    L_R = np.delete(L_R, 0)                 # Taking out first
    L_R = np.delete(L_R, -1)                # Taking out last
    L_R = L_R.astype(int)
    
    return [L_B, L_R]
    
# Right boundary
def Right_BR(Nx, Ny):
    
    # Right boundary
    L_B = np.linspace(Nx, (Ny - 2) * Nx, Ny - 2)
    R_B = L_B + Nx - 1
    R_B = np.transpose(R_B)
    R_B = R_B.astype(int)
    del(L_B)
    
    R_R = R_B - 1                           # Right ring
    R_R = np.delete(R_R, 0)                 # Taking out first
    R_R = np.delete(R_R, -1)                # Taking out last
    R_R = R_R.astype(int)
   
    return [R_B, R_R]

# ==============================================================================
# IMPOSING INITIAL CONDITION
# ==============================================================================

def I_C(X, Y, vzero):
    
    # Initial conditions for Burgers 2D test equation - use just one and set the 
    # other one to 0 for testing purposes
    
    # Velocity equals zero in y direction
    if vzero == 'v':
        u0 = -np.sin(np.pi * X)
        v0 = 0 * X * Y

    # Velocity equals 0 in x direction
    elif vzero == 'u':        
        u0 = 0 * X * Y
        v0 = -np.sin(np.pi * Y)
    
    return [u0, v0]

# ==============================================================================
# ASSEMBLYING MATRICES FOR VISCOUS TERM IN BOTH DIRECTIONS (assuming boundary
# conditions given in the arrays of BC)
# ==============================================================================
    
def Ass_matrix(K, Nx, Ny, Sx, Sy, BC0, Der2):
    
    # Defining matrix according to number of nodes in the domain
    nn = Nx * Ny
    
    # Looking for boundary nodes in the domain according to Nx and Ny
    [B_B, B_R] = Bot_BR(Nx, Ny)
    [T_B, T_R] = Top_BR(Nx, Ny)
    [L_B, L_R] = Left_BR(Nx, Ny)
    [R_B, R_R] = Right_BR(Nx, Ny)    
            
    # Filling matrix with typical nodal values (then the boundary nodes are 
    # replaced with rows equal to 0 and changed) - this works for both implemen-
    # tations of the second derivative
    
    if Der2 == 0:
        
        for i in range(int(B_R[0]), int(T_B[0])):
            
            # Second order second derivative for the matrix
            K[i, i] = 1 + 2 * Sx + 2 * Sy
            K[i, i + 1] = -Sx
            K[i, i - 1] = -Sx
            K[i, i + Nx] = -Sy
            K[i, i - Nx] = -Sy
            
    elif Der2 == 1:
        
        # Coefficients of matrix change in higher order
        Sx1 = Sx / 12
        Sy1 = Sy / 12
        
        for i in range(L_B[1] + 1, R_B[-2] - 1):
            
            # Fourth order derivative for the viscous calculation
            K[i, i] = 1 + 30 * Sx1 + 30 * Sy1
            K[i, i + 1] = -16 * Sx1
            K[i, i - 1] = -16 * Sx1
            K[i, i + Nx] = -16 * Sy1
            K[i, i - Nx] = -16 * Sy1
            K[i, i + 2] = Sx1
            K[i, i - 2] = Sx1
            K[i, i + 2 * Nx] = Sy1
            K[i, i - 2 * Nx] = Sy1           
            
        # Setting the value of the rows that represent external ring to 0
        K[B_R, :] = np.zeros(nn)
        K[T_R, :] = np.zeros(nn)
        K[R_R, :] = np.zeros(nn)
        K[L_R, :] = np.zeros(nn)
        
        # Setting the value of the matrix elements in the external ring
        K[B_R, B_R] = 1 + 2 * Sx + 2 * Sy
        K[B_R, B_R + 1] = -Sx
        K[B_R, B_R - 1] = -Sx
        K[B_R, B_R + Nx] = -Sy
        K[B_R, B_R - Nx] = -Sy
        
        K[T_R, T_R] = 1 + 2 * Sx + 2 * Sy
        K[T_R, T_R + 1] = -Sx
        K[T_R, T_R - 1] = -Sx
        K[T_R, T_R + Nx] = -Sy
        K[T_R, T_R - Nx] = -Sy
        
        K[R_R, R_R] = 1 + 2 * Sx + 2 * Sy
        K[R_R, R_R + 1] = -Sx
        K[R_R, R_R - 1] = -Sx
        K[R_R, R_R + Nx] = -Sy
        K[R_R, R_R - Nx] = -Sy
        
        K[L_R, L_R] = 1 + 2 * Sx + 2 * Sy
        K[L_R, L_R + 1] = -Sx
        K[L_R, L_R - 1] = -Sx
        K[L_R, L_R + Nx] = -Sy
        K[L_R, L_R - Nx] = -Sy
        
    # Setting the value of the rows that represent boundary elements to 0 - the 
    # same for both constructions
    K[B_B, :] = np.zeros(nn)
    K[T_B, :] = np.zeros(nn)
    K[R_B, :] = np.zeros(nn)
    K[L_B, :] = np.zeros(nn)
    
    # Setting boundary elements - it is the same for both matrix constructions
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

# ==============================================================================
# FIRST DERIVATIVES IN SPACE. EACH CASE OF DERIVATION IS TREATED HERE
# THE FLAG THAT MARKS THE TYPE OF DERIVATION IS THE VARIABLE dift
# ==============================================================================
# ==============================================================================
# Differentiating a vector in the x direction - not imposing BC yet. Works for 
# u and v independently
# ==============================================================================
    
def diffx(u, dx, L_B, L_R, R_B, R_R, dift):
    
    # Parameters that can be changed for convenience when running the function
    q = 0.5
    
    # Defining vector that will store the derivative - column vector, like the
    # one that enters the function
    diffx = np.zeros((len(u), 1))
    nn = len(u)
    u = u.reshape((len(u), 1))
    
    # Normal upwind direction
    if dift == 0:
        
        for i in range(1, nn - 1):
            
            if u[i] >= 0:
                
                diffx[i] = (u[i] - u[i - 1]) / dx
            
            else:
                
                diffx[i] = (u[i + 1] - u[i]) / dx
        
        diffx[L_B] = (u[L_B + 1] - u[i]) / dx
        
        diffx[R_B] = (u[i] - u[R_B - 1]) / dx
    
    # Corrected three point upwind from Fletcher 1991    
    elif dift == 1:
        
        # Appending nodes to the boundary and ring arrays to count every possible 
        # scenario
        Nx = L_B[0]
        L_B1 = np.append(0, L_B)
        L_B1 = np.append(L_B1, L_B[-1] + Nx)
        L_R1 = np.append(np.array((L_R[0] - 2 * Nx, L_R[0] - Nx)), L_R)
        L_R1 = np.append(L_R1, np.array((L_R[-1] + Nx, L_R[-1] + 2 * Nx)))
        R_B1 = np.append(R_B[0] - Nx, R_B)
        R_B1 = np.append(R_B1, R_B[-1] + Nx)
        R_R1 = np.append(np.array((R_R[0] - 2 * Nx, R_R[0] - Nx)), R_R)
        R_R1 = np.append(R_R1, np.array((R_R[-1] + Nx, R_R[-1] + 2 * Nx)))
            
        for i in range(0, nn):
                        
            # Testing if element is in Left boundary - since it is a boundary I 
            # have to apply first order upwind scheme
            if np.isin(i, L_B1): 
                
                diffx[i] = (u[i + 1] - u[i]) / dx 
                    
            # Testing if element is in Left ring - Second order is applied only 
            # if velocity is negative
            elif np.isin(i, L_R1):
                
                if u[i] >= 0 : diffx[i] = (u[i] - u[i - 1]) / dx
                
                else:
                    
                    diffx[i] = (u[i + 1] - u[i - 1]) / (2 * dx) + q * (u[i - 1]\
                         - 3 * u[i] + 3 * u[i + 1] - u[i + 2]) / (3 * dx)
                    
            # Testing if element is in right boundary - First order upwind since
            # it is a boundary
            elif np.isin(i, R_B1): 
                
                diffx[i] = (u[i] - u[i - 1]) / dx
                    
            # Testing if element is in right ring - Second order is applied only
            # if velocity is positive
            elif np.isin(i, R_R1):
                
                if u[i] < 0 : diffx[i] = (u[i + 1] - u[i]) / dx
                    
                else:
                    
                    diffx[i] = (u[i + 1] - u[i - 1]) / (2 * dx) + q * (u[i - 2]\
                         - 3 * u[i - 1] + 3 * u[i] - u[i + 1]) / (3 * dx)
                                
            # Internal nodes - The choice is made according to the velocity and 
            # the high order scheme can be applied
            else:
                
                if u[i] >= 0:
                    
                    diffx[i] = (u[i + 1] - u[i - 1]) / (2 * dx) + q * (u[i - 2]\
                         - 3 * u[i - 1] + 3 * u[i] - u[i + 1]) / (3 * dx)              
                    
                else:
                    
                    diffx[i] = (u[i + 1] - u[i - 1]) / (2 * dx) + q * (u[i - 1]\
                         - 3 * u[i] + 3 * u[i + 1] - u[i + 2]) / (3 * dx)
    
    return diffx

# ==============================================================================
# Differentiating a vector in the y direction - not imposing BC yet. Works for
# u and v independently
# ==============================================================================

def diffy(v, dy, B_B, B_R, T_B, T_R, dift):
    
    # Parameters that can be changed for convenience when running the function
    q = 0.5
    
    # Defining vector that will store the derivative - column vector like the 
    # one that enters to the function
    nn = len(v)
    diffy = np.zeros((len(v), 1))
    Nx = len(B_B)
    v = v.reshape((len(v), 1))
    
    # Normal upwind direction
    if dift == 0:
        
        for i in range(B_B[-1] + 1, T_B[0]):
            
            if v[i] >= 0:
                
                diffy[i] = (v[i] - v[i - Nx]) / dy
            
            else:
                
                diffy[i] = (v[i + Nx] - v[i]) / dy
                
        diffy[B_B] = (v[B_B + Nx] - v[B_B]) / dy
        
        diffy[T_B] = (v[T_B] - v[T_B - Nx]) / dy
        
    elif dift == 1:
        
        # Appending nodes to the boundary and ring arrays to count every possible 
        # scenario
        B_R1 = np.append(B_R[0] - 1, B_R)
        B_R1 = np.append(B_R1, B_R[-1] + 1)
        T_R1 = np.append(T_R[0] - 1, T_R)
        T_R1 = np.append(T_R1, T_R[-1] + 1)   
        
        for i in range(0, nn):
            
            # Testing if element is in Bottom boundary - since it is a 
            # boundary I have to apply first order upwind scheme
            if np.isin(i, B_B) : diffy[i] = (v[i + Nx] - v[i]) / dy 
                    
            # Testing if element is in Bottom ring - Second order is applied 
            # only if velocity is negative
            elif np.isin(i, B_R1):
                
                if v[i] >= 0 : diffy[i] = (v[i] - v[i - Nx]) / dy
                
                else:
                    
                    diffy[i] = (v[i + Nx] - v[i - Nx]) / (2 * dy) + q * \
                    (v[i - Nx] - 3 * v[i] + 3 * v[i + Nx] - v[i + 2 * Nx]) \
                    / (3 * dy)
                    
            # Testing if element is in top boundary - First order upwind since
            # it is a boundary
            elif np.isin(i, T_B) : diffy[i] = (v[i] - v[i - Nx]) / dy
                    
            # Testing if element is in top ring - Second order is applied only
            # if velocity is positive
            elif np.isin(i, T_R1):
                
                if v[i] < 0 : diffy[i] = (v[i + Nx] - v[i]) / dy
                    
                else:
                    
                    diffy[i] = (v[i + Nx] - v[i - Nx]) / (2 * dy) + q * \
                    (v[i - 2 * Nx] - 3 * v[i - Nx] + 3 * v[i] - v[i + Nx]) \
                    / (3 * dy)
                                
            # Internal nodes - The choice is made according to the velocity and 
            # the high order scheme can be applied
            else:
                
                if v[i] >= 0:
                    
                    diffy[i] = (v[i + Nx] - v[i - Nx]) / (2 * dy) + q * \
                    (v[i - 2 * Nx] - 3 * v[i - Nx] + 3 * v[i] - v[i + Nx]) \
                    / (3 * dy)              
                    
                else:
                    
                    diffy[i] = (v[i + Nx] - v[i - Nx]) / (2 * dy) + q * \
                    (v[i - Nx] - 3 * v[i] + 3 * v[i + Nx] - v[i + 2 * Nx]) \
                    / (3 * dy)
    
    return diffy

# ==============================================================================
