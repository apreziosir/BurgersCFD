#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routine that solves the 1D Burgers' equation
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
May 2018
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import style
import Analytical as an
import Non_linear as nl


# ==============================================================================
# Physical variables declaration
# ==============================================================================

X0 = -1
XF = 1
nu = 1e-2
t0 = 0.
tf = 2 / np.pi 

# ==============================================================================
# Numerical parameters declaration
# ==============================================================================

# Management of the non linear term:    0. Primitive variables, 
#                                       1. Divergence form
#                                       2. Skew symmetric form
nlt = 0 

# Number of nodes in the domain (changeable for convergence analysis)
N = 500

# Maximum CFL number
CFL = 1.

# Generating vector of nodes
xn = np.linspace(X0, XF, N)

# Calculating dx for value between nodes
dx = xn[1] - xn[0]

# Estimating initial condition
u0 = -np.sin(np.pi * xn)

# Estimating timestep size dt
dt = (np.max(np.abs(u0)) * dx) / CFL

# Calculating the number of timesteps of the model
nT = int(np.ceil((tf - t0) / dt))

# Generating vector of errors
ert = np.zeros(nT)

# ==============================================================================
# Assembling sparse matrix for viscous term solution. Efficient storage of 
# matrix for memory purposes
# ==============================================================================

# Defining matrix
K = sp.lil_matrix((N, N))

# Matrix coefficients - for high precision and second to last nodes
Sxp = nu * dt / (12 * dx ** 2)
Sx = nu * dt / (dx ** 2)

# Matrix coefficients for Dirichlet boundary conditions
K[0, 0] = 1
K[N - 1, N - 1] = 1

# Matrix coefficients for second and second to last elements
K[1, 1] = 1 + 2 * Sx 
K[1, 0] = -Sx
K[1, 2] = -Sx

K[N - 2, N - 2] = 1 + 2 * Sx
K[N - 2, N - 1] = -Sx
K[N - 2, N - 3] = -Sx

# Matrix coefficients for the high precision part of the domain
for i in range(2, N - 2):
    
    K[i, i] = 1 + 30 * Sxp
    K[i, i + 1] = -16 * Sxp
    K[i, i - 1] = -16 * Sxp
    K[i, i + 2] = Sxp
    K[i, i - 2] = Sxp

K = K.tocsr() 

# ==============================================================================
# Imposing initial condition as given in the problem
# ==============================================================================

ug = np.zeros(len(u0))
u1 = np.zeros(len(u0))

ert[0] = 0

umax = np.max(u0)
umin = np.min(u0)

# Plotting initial condition
plt.ion()
plt.figure(1, figsize=(11, 8.5))
style.use('ggplot')

plt.subplot(1, 1, 1)
plt.plot(xn, u0)
plt.title('Initial condition')
plt.xlabel(r'Distance $(m)$')
plt.ylabel(r'Velocity $ \frac{m}{s} $')
plt.draw()
plt.pause(1.5)

# ==============================================================================
# Startting time loop of the program (solved via fractional steps)
# ==============================================================================

for t in range(1, nT):
    
    # Calculating the analytical solution for the given timestep
    ua = an.Analyt(xn, t * dt, nu)
    
    # Calculating the non linear term using one of the three possibilities
    # The nlt variable is defined in the numerical parameters of the program in 
    # order to check differences
    if nlt == 0:
        
        # Primitive variables
#        print('Entre a variables primitivas')
        ug = nl.prim_var(u0, dx, dt)
        
    elif nlt == 1:
        
        # Divergence form
#        print('Entre a divergencia')
        ug = nl.div_f(u0, dx, dt)
        
    else:
        
        # Skew symmetric form
#        print('Entre a Skew-symmetric')
        ug = nl.Skew_sym(u0, dx, dt)
        
    # Calculating the diffusive term with high order derivatives in the internal
    # nodes (5 points approximation)
    
    # Imposing boundary conditions - Homogeneous Dirichlet
    ug[0] = 0
    ug[N - 1] = 0
    
    u1 = spsolve(K, ug)
    
    # Calculating error
    err = np.abs(u1 - ua)
    ert[t + 1] = np.linalg.norm(err)
    
    # Plotting numerical solution and comparison with analytical
    plt.clf()
    
    plt.subplot(2, 2, 1)
    plt.plot(xn, u1, 'b')
    plt.xlim([X0, XF])
    plt.ylim([umin, umax])
    plt.ylabel(r'Velocity $ \frac{m}{s} $')
    plt.title('Numerical solution')
    
    plt.subplot(2, 2, 2)
    plt.plot(xn, ua)
    plt.xlim([X0, XF])
    plt.ylim([umin, umax])
    plt.title('Analytical solution')
    
    plt.subplot(2, 2, 3)
    plt.semilogy(xn, err)
    plt.xlim([X0, XF])
    plt.ylim([1e-4, 1e2])
    plt.xlabel('Error in space')
    plt.ylabel('Absolute error')
    plt.title('Error')
    
    plt.subplot(2, 2, 4)
    plt.semilogy(np.linspace(t0, tf, nT), ert)
    plt.xlim([t0 - 0.2, tf + 0.2])
    plt.ylim([1e-4, 1e2])
    plt.xlabel(r'Time $ (s) $')
    plt.title('Error evolution')
    
    plt.draw()
    titulo = '1D Burgers equation'
    plt.suptitle(titulo)
    plt.pause(0.01)
    
#    if ts[t] == 2 / np.pi:
#        
#        plt.svaefig('ejemplo.pdf')
        
    
    # Updating velocities for next timestep
    u0 = u1
    
    