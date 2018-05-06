#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routine that solves the 1D Burgers' equation
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
May 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import Analytical as an


# ==============================================================================
# Physical variables declaration
# ==============================================================================

X0 = -1
XF = 1
nu = 0.1

# ==============================================================================
# Numerical parameters declaration
# ==============================================================================

# Number of nodes in the domain (changeable for convergence analysis)
N = 101

# Generating vector of nodes
xn = np.linspace(X0, XF, N)

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