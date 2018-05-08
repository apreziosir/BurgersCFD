#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that compares the results from the burgers' 1d equation, with
different spatial and temporal discretizations
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
"""


# Importing the functions that calculate with different discretizations
from Burgers1D_f import Burgers_f


# Importing libraries to plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


# Declaring variables to be run and then plotted
nT = np.array([5, 10, 20, 50, 75, 100, 200, 500, 1000, 2000, 5000])
N = np.array([5, 10, 20, 50, 75, 100, 500, 1000, 2000, 5000])
nlt = np.array([0, 1, 2])

# Matrix to store maximum errors of each run
MD = np.zeros((np.size(nT), np.size(N), np.size(nlt)))

# Looping to find maximum error of each run
for i in range(0, np.size(nT)):
    
    for j in range(0, np.size(N)):
        
        for k in range(0, np.size(nlt)):
            
            MD[i, j, k] = np.linalg.norm(Burgers_f(N[j], nT[i], nlt[k]))


for II in range(0, 3):
    
    # Plotting the error curves for each Sx
    plt.figure(II, figsize=(11, 8.5))
    style.use('ggplot')        
    
    plt.subplot(1, 2, 1)
    for i in range(0, len(N)):
        line1 = plt.loglog(nT, MD[:, i, II], label= "N = " + str(N[i]))
    #    plt.xlim([np.min(Nx), np.max(Nx)])
        plt.ylim([1e-7, 2e-1])
        plt.gca().invert_xaxis()
        plt.ylabel(r'Infinity error norm')
        plt.xlabel('Timestep size (s)')
        plt.legend(loc=4)
        plt.title('Time refining error evolution') 
        plt.gca().invert_xaxis()
    
    #
    for i in range(0, len(nT)):
        plt.subplot(1, 2, 2)
        line1 = plt.semilogy(N, MD[i,:, II], label = 'dT = ' + str(nT[i]))
        plt.xlim([np.min(N), np.max(N)])
        plt.ylim([1e-7, 2e-1])
        plt.xlabel('Number of nodes')
        plt.legend(loc=1)
        plt.title('Space refining error evolution')
    #
    
    plt.suptitle('Error analysis for 1D Burgers with nlt = ' + str(II))
    plt.draw()
    plt.savefig('Convergence' + str(II) + '.pdf')