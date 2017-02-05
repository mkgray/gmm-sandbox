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
import collections

if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    
    # Set up potential arguments for parsing
    parser.add_argument("-s", "--samples", type=int, help="number of samples to generate", default=10000)
    parser.add_argument("-d", "--dimensions", type=int, help="number of dimensions to generate", default=2)
    parser.add_argument("-m", "--models", type=int, help="number of gaussian models in mixture", default=3)
    args = parser.parse_args()
    
    # Spit out setting to user
    print("Gaussian Mixture Model for generation:")
    print("Number of samples: " + str(args.samples))
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
        gaussian_model_covar[:,:,i] = np.dot(gaussian_model_covar[:,:,i],gaussian_model_covar[:,:,i].T) # Force positive semidefinite
        gaussian_model_covar[:,:,i] = (gaussian_model_covar[:,:,i] + gaussian_model_covar[:,:,i].T)/2 # Force symmetrical
    
    # Generate the gmm data
    # For each sample, determine the Gaussian model to draw from
    gaussian_model_samples = np.random.choice(np.arange(0, args.models), p=normalized_model_weights, size=args.samples)
    
    # Accumulate the model samples to batch generate from each gaussian model for faster generation
    model_occurrences = collections.Counter(gaussian_model)
    
    # Initialize the empty array to store all values
    complete_dataset = np.empty([args.samples,args.dimensions])
    complete_labels = np.empty([args.samples,1])    
    
    # Generate the points from each model
    fill_index = 0
    for i in model_occurrences:
        # Step 1: Load the model parameters
        model_mean = gaussian_model_means[:,i]
        model_covariance = np.squeeze(gaussian_model_covar[:,:,i])
        
        # Step 2 Sample the parameters
        complete_dataset[fill_index:(fill_index+model_occurrences[i]),:] = np.random.multivariate_normal(model_mean, model_covariance, model_occurrences[i])
        complete_labels[fill_index:(fill_index+model_occurrences[i]),0] = i
        fill_index += model_occurrences[i]
    
    '''    
    # Plot the points for visualization
    plt.scatter(x, y)
    plt.show
    '''