# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 20:27:01 2015

@author: Andrew
"""
import os; os.getcwd(); os.walk('.') # See what directory you are in
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from helperFunctions import plotData, sigmoid, mapFeature, costFunctionDRW, log2

## Machine Learning - Exercise 2: Logistic Regression

def predict(theta, X):
    """
    Return predictions for a set of test scores.
    """

    p = sigmoid(np.dot(X_array, theta)) >= 0.5
    return p

if __name__ == '__main__':
    ## Initialization
    plt.close('all')

    ## Load Data
    #  The first two columns contains the exam scores and the third column
    #  contains the label.
    data = pd.read_csv('ex2data1.txt', header=None)
    X = data[[0, 1]]
    y = data[2]
    y = y + 0.0

    ## ==================== Part 1: Plotting ====================
    #  We start the exercise by first plotting the data to understand the
    #  the problem we are working with.

    print 'Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n'

    figure = plotData(X, y)


    ## ============ Part 2: Compute Cost (w/o gradient) ============
    
    #  Setup the data matrix appropriately, and add ones for the intercept term
    m, n = np.shape(X)

    # Add intercept term to x and X_test
    X_array = mapFeature(X)
    y_array = np.array(y)

    # Initialize fitting parameters
    initial_theta = np.zeros(n + 1)

    # Compute and display initial cost and gradient
    cost = costFunctionDRW(initial_theta, X_array, y_array)

    print 'Cost at initial theta (zeros): %f' % cost


    ## ============= Part 3: Optimizing using DRW =============
    # From scipy.optimize, the minimize function looks to offer similar functionality
    # to fminunc.
    
    # Using Nelder-Mead
    result_Nelder_Mead = minimize(lambda t: costFunctionDRW(t, X_array, y_array), x0=initial_theta, method='nelder-mead', options = {'maxiter':1000})
      
    print 'Cost at theta found by Nelder-Mead: %f' % result_Nelder_Mead['fun']
    print 'theta:', result_Nelder_Mead['x']
    theta = result_Nelder_Mead['x']

    # Plot Boundary
    plotData(X, y, theta)
    

    
    ## ============== Part 4: Predict and Check Accuracy ==============
    #  Predict probability for a student with score 45 on exam 1
    #  and score 85 on exam 2
    prob = sigmoid(np.dot(np.array([1, 45, 85]), theta))
    print 'For a student with scores 45 and 85, we predict an admission probability of',
    print '%.1f%%' % (prob * 100)

    # Compute accuracy on our training set
    p = predict(theta, X_array)
    print 'Train Accuracy: %.1f%%' % ((p == y_array).mean() * 100)
    
    # Calculate missing information (i.e. entropy)
    m = len(y)
    marg_pA = sum(y)/m
    marg_pR = 1 - marg_pA
    entropy = marg_pA*log2(1/marg_pA) + marg_pR*log2(1/marg_pR)
    print 'Entropy: %.4f bits' % entropy
    
    # Compute the doubling rate of wealth (i.e. information gain)
    empirical_wealth = -costFunctionDRW(theta, X_array, y_array)
    log_wealth = log2(empirical_wealth)
    drw = (1/len(y_array))*log_wealth 
    print 'Doubling rate: %.4f bits' % drw
    
    # Compute information gain as a percentage
    info_gain = (drw/entropy)*100
    print 'Info gain: %.1f%%' % info_gain
