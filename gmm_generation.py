# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 09:57:15 2017

@author: Matthew
"""

# Calculate divisions as floats naturally
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    
    # Set up potential arguments for parsing
    parser.add_argument("-d", "--dimensions", type=int, help="number of dimensions to generate", default=2)
    parser.add_argument("-m", "--models", type=int, help="number of gaussian models in mixture", default=3)
    args = parser.parse_args()
    
    # Spit out setting to user
    print("Gaussian Mixture Model for generation:")
    print("Number of dimensions: " + str(args.dimensions))
    print("Number of Gaussian models in mixture: " + str(args.models))
    
    # Gaussian Mixture Models (GMM) require a weighting vector
    # to establish probability of selection frome each Gaussian
    # model
    raw_model_weights = np.random.rand(1,args.models)
    normalized_model_weights = np.squeeze(raw_model_weights/np.sum(raw_model_weights))
    
    # Initialize parameters for each Gaussian model
    gaussian_model_means = np.squeeze(np.random.rand(args.dimensions,args.models)*20-10)
    # Covariance matrices must be positive semidefinite and symmetrical
    gaussian_model_covar = np.squeeze(np.random.rand(args.dimensions,args.dimensions,args.models)*3-1.5)
    
    # Force covariances to be symmetrical
    for i in range(gaussian_model_covar.shape[-1]):
        gaussian_model_covar[:,:,i] = (gaussian_model_covar[:,:,i] + gaussian_model_covar[:,:,i].T)/2
    
    # Generate the gmm data
    # For each point, determine the Gaussian model to draw from
    gaussian_model = np.random.choice(np.arange(0, args.models), p=normalized_model_weights)
    
    # Pull the parameters of the Gaussian model (mu and sigma) for each dimension
    model_mean = gaussian_model_means[:,gaussian_model]
    model_covar = np.squeeze(gaussian_model_covar[:,:,gaussian_model])
    
    # Generate the d-dimensional point
    x, y, z = np.random.multivariate_normal(model_mean, model_covar, 100).T
    
    # Plot the points for visualization
    plt.scatter(x, y)
    plt.show