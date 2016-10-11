#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:36:22 2016

@author: shazz
"""

# The usual preamble
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

def load_data(row_nb):
    na_values = ['    ']
    data = pd.read_csv("data/2015_BRFSS_extract.tsv", sep='\t', header=0, na_values=na_values)
    
    print("data loaded:", data.shape)
    
    # remove non sense heights
    overm_heights = data['HEIG'] >= 800
    overl_heights = data['HEIG'] == 0
    data['HEIG'][overm_heights] = np.nan
    data['HEIG'][overl_heights] = np.nan
    
    # remove non sense weights
    overm_weights = data['WEIG'] >= 600
    overl_weights = data['WEIG'] == 0
    data['WEIG'][overm_weights] = np.nan
    data['WEIG'][overl_weights] = np.nan

    data = data.dropna()

    if row_nb > 0:
        data = data.iloc[:row_nb, :]
    
    X = data[["WEIG","HEIG"]]   
    Y = np.ravel(data[["S"]])
    
    #print("Genre values:", data['S'].unique())
    #print("Height values:", data['HEIG'].unique())
    #print("Weight values:", data['WEIG'].unique())

    return X, Y