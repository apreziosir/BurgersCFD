#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Burgers 2D solver with FDM. 
Implementation of different forms of treatment of the non linear term (primitive
variables, divergence form and skew symmetric form).
Fractional timestep implemented with explicit solution for advective term and 
implicit solution of the diffusive term
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana, Bogotá
May 2018
"""

# Modules are commented to see if they have influence on the construction of 
# plots (they can be removed safely at the end when the program is running)
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#from pylab import *
from matplotlib import style
import Auxiliary2D as AUX
import Analytical2D as an
#import Non_linear_2D as nl

# ==============================================================================
# DECLARATION OF PHYSICAL VARIABLES
# ==============================================================================

# Physical boundaries of the domain
X0 = -1.
XF = 1.
Y0 = -1.
YF = 1.

# Real time simulated in the model (for Burgers the final time is 2 / pi)
t0 = 0
tf = 2 / np.pi 

# Physical variables of the fluid - mainly viscosity (include density when 
# dealing with N-S solver)
nu_x = 1e-2
nu_y = 1e-2

# ==============================================================================
# DECLARATION OF NUMERICAL PARAMETERS
# ==============================================================================

# Treatment of the non linear term. 0 = primitive variables, 1 = divergence form
# 2 = skew symmetric form
nlt = 0

# Maximum CFL values for each dimension
CFL_x = 1.0
CFL_y = 1.0

# Number of nodes in each direction
Nx = 9                                         # Nodes in x direction
Ny = 9                                         # Nodes in y direction
nn = Nx * Ny                                    # Number of nodes (total)

# Boundary conditions vectors BC0_u = type of BC. BC1_u = value of the BC. 
# BC0_v = type of BC, BC1_v = value of the BC. 
# The order of the BC is [bottom, top, left, right]
# For BC0 0 = Dirichlet, 1 = Neumann
# For BC1 the value of the Bc is stored
BC0_u = np.array((1, 1, 0, 0))
BC0_v = np.array((0, 0, 1, 1))

BC1_u = np.array((0, 0, 0, 0))
BC1_v = np.array((0, 0, 0, 0)) 

# Generating the spatial vectors - and then the meshgrid for plotting
xn = np.linspace(X0, XF, Nx)
yn = np.linspace(Y0, YF, Ny)

# Calculating dx and dy
dx = np.absolute(xn[1] - xn[0])
dy = np.absolute(yn[1] - yn[0])

[X, Y] = np.meshgrid(xn, yn)

# ==============================================================================
# INITIAL CONDITION CALCULATION FOR ESTIMATION OF THE TIMESTEP SIZE
# ==============================================================================

# Imposing initial condition - in both dimensions
[u0, v0] = AUX.I_C(X, Y)

# Estimating the size of the timestep according to maximum velocity. Checking 
# and comparing each of the maximum velocities. 
umax = np.amax(np.amax(np.absolute(u0)))
vmax = np.amax(np.amax(np.absolute(v0)))
UVmax = np.maximum(umax, vmax)

# Calculating the timestep size according to CFL and maximum velocity in each 
# one of the directions
dT = UVmax * np.maximum(dx, dy) / np.maximum(CFL_x, CFL_y)

# Calculating the number of timesteps of the model
nT = int(np.ceil((tf - t0) / dT))

# ==============================================================================
# SEARCHING THE NODES THAT CORRESPOND TO THE BOUNDARY OF THE DOMAIN - This 
# makes easier the implementation of the boundary conditions in the matrices
# Also looking for nodes in the first ring adjacent to boundary for making 
# higher order approximation in viscous term (function outside main to be 
# called)
# ==============================================================================

# Boundaries
[B_B, B_R] = AUX.Bot_BR(Nx, Ny)
[T_B, T_R] = AUX.Top_BR(Nx, Ny)
[L_B, L_R] = AUX.Left_BR(Nx, Ny)
[R_b, R_R] = AUX.Right_BR(Nx, Ny)

# ==============================================================================
# Constructing the Stiffness Matrices for the viscous terms. I need two matrices
# one for each one of the velocity fields (K_x and K_y)
# ==============================================================================

# Estimating value of the nondimensional parameters for the stiffness matrix
# They will go as an input to the function that assembles the stiffness matrices
Sx = nu_x * dT / (dx ** 2)
Sy = nu_y * dT / (dy ** 2)

# Staring up both stiffness matrices for u and v velocities (no values yet, just 
# sparse and python efficiently saved)
K_x = sp.lil_matrix((nn, nn))
K_y = sp.lil_matrix((nn, nn))

# Calling function to fill matrices according to BC type 
K_x = AUX.Ass_matrix(K_x, Nx, Ny, Sx, Sy, BC0_u)
K_y = AUX.Ass_matrix(K_y, Nx, Ny, Sx, Sy, BC0_v)

# Storing in CSR format for implementing efficient solving in the time loop
K_x = K_x.tocsr()
K_y = K_y.tocsr()

# ==============================================================================
# Plotting initial conditions for the case studied
# ==============================================================================

style.use('ggplot')
plt.figure(1, figsize=(20, 15))

fig1 = plt.subplot(1, 2, 1, projection='3d')
surf1 = fig1.plot_surface(X, Y, u0, rstride=1, cstride=1, linewidth=0, 
                       cmap=cm.coolwarm, antialiased=False)
plt.title('Initial condition for U (m/s)')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')


fig2 = plt.subplot(1, 2, 2, projection='3d')
surf1 = fig2.plot_surface(X, Y, v0, rstride=1, cstride=1, linewidth=0, 
                       cmap=cm.coolwarm, antialiased=False)
plt.title('Initial condition for V (m/s)')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

plt.pause(3)

# ==============================================================================
# Reshaping matrices that sotre results to vectors in order to make them solva-
# ble by matrix-vector operations
# ==============================================================================


#U = an.Analyt(X, Y, 0.2, nu_x)
