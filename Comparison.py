#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that compares the results from the burgers' 1d equation, with
different spatial and temporal discretizations
@author: Antonio Preziosi-Ribero
CFD - Pontificia Universidad Javeriana
"""


# Importing the functions that calculate with different discretizations
from Burgers1D_f import Burg1D

#Burg1D(nu, nlt, N)


# Importing libraries to plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


# Declaring variables to be run and then plotted
nu = np.array([1e-2, 5e-2, 1e-1, 1])
nlt = np.array([0, 1, 2])
N = np.array([21, 31, 41, 51, 81, 101, 251, 501, 1001, 2001])

# Matrix to store maximum errors of each run
MD = np.zeros((np.size(nu), np.size(nlt), np.size(N)))

# Looping to get the different results

for i in range(0, len(nu)):
    
    for j in range(0, len(nlt)):
        
        for k in range(0, len(N)):            
            
            temp = Burg1D(nu[i], 1, N[k])
            MD[i, j, k] = np.linalg.norm(temp)
            
            
# Plotting the different results in order to compare them 
            
style.use('ggplot')    

for II in range(0, len(nlt)):

    plt.figure(II, figsize=(11, 8.5))

    plt.subplot(1,1,1)
    for i in range(0, len(nu)):
        line1 = plt.loglog(N, MD[i, II, :], label=r'$ \nu = $' + str(nu[i]))
        plt.xlim([np.min(N), np.max(N)])
        plt.ylim([1e-1, 1e2])
        plt.xlabel('Number of nodes')
        plt.ylabel(r'Infinity error norm')
        plt.legend(loc=4)
        plt.title('Error with different viscosities')
        
        
    
        
    

