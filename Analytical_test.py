#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 23:45:06 2018
Testing analytical solution behavior
@author: toni
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import Analytical as an

# Physical conditions
x = np.linspace(-1, 1, 101)
nu = 0.01

# Initial condition
u0 = -np.sin(np.pi * x)

dt = 0.01
nT = 50

plt.ion()
plt.figure(1, figsize=(11, 8.5))
style.use('ggplot')

for i in range(0, nT):
    
    t = (i + 1) * dt
    u = an.Analyt(x, t, nu)
    
    plt.clf()
    plt.subplot(1, 1, 1)
    plt.plot(x, u)
    plt.title('Analytical evolution')
    plt.xlabel(r'Distance $(m)$')
    plt.ylabel(r'Velocity $ \frac{m}{s} $')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.draw()
    plt.pause(0.2)