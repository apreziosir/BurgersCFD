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
import matplotlib.pyplot as plt
from matplotlib import style
import Analytical as an
import Non_linear as nl


# ==============================================================================
# Physical variables declaration
# ==============================================================================

X0 = -1
XF = 1
nu = 0.1
t0 = 0
tf = 3

# ==============================================================================
# Numerical parameters declaration
# ==============================================================================

# Number of nodes in the domain (changeable for convergence analysis)
N = 101

# Number of timesteps for the calculation
nT = 2000

# Calculating dx for value between nodes
dx = (XF - X0) / (N - 1)

# Generating vector of nodes
xn = np.linspace(X0, XF, N)

# Generating vector of times
ts = np.linspace(t0, tf, nT) / np.pi 

# Calculating timestep size from timestep vector
dt = ts[1] - ts[0]

# ==============================================================================
# Assembling sparse matrix for viscous term solution. Efficient storage of 
# matrix for memory purposes
# ==============================================================================

# Defining matrix
K = sp.lil_matrix((N, N))



# ==============================================================================
# Imposing initial condition as given in the problem
# ==============================================================================

u0 = -np.sin(np.pi * xn)

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

for t in range(1, len(ts)):
    
    # Calculating the non linear term using one of the three possibilities
    
    
    # Calculating the diffusive term with high order derivatives in the internal
    # nodes (5 points approximation)
    
    