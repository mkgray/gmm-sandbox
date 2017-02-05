# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 09:57:15 2017

@author: Matthew
"""

# Calculate divisions as floats naturally
from __future__ import division

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
    
    # Generate the gmm data
    