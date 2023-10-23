#!/usr/bin/env python
# coding: utf-8

# # Function

import os

import numpy as np
import statistics
import pyedflib
# import mne
import math


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from IPython.display import display, Markdown  #display(Markdown("# Hello World!"))


# ## Read edf file
def readedf(path):
    f = pyedflib.EdfReader(path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i,:] = f.readSignal(i)
    return sigbufs


# ## Feature Calculate

def feat_mean(array):
    return np.mean(array)
def feat_variance(array):
    return np.var(array, axis=0)
def feat_skewness(array):
    skewness = np.mean((array - np.mean(array))**3) / np.std(array)**3
    return skewness
def feat_kurtosis(array):
    kurtosis = np.mean((array - np.mean(array))**4) / np.std(array)**4
    return kurtosis
def feat_cov(array):
    CoV = (np.std(array)/np.mean(array))
    return CoV
def feat_mad(array):
    median = np.median(array)
    abs_deviation = np.abs(array-median)
    mad = np.median(abs_deviation)
    return mad
def feat_rms(array):
    rms_amplitude = np.sqrt(np.mean(np.square(array)))
    return rms_amplitude
def feat_shannon_entropy(sequence):
    uniqw, inverse = np.unique(sequence, return_inverse=True)
    event_counts = np.bincount(inverse)

    # Calculate probabilities
    total_events = len(sequence)
    event_probabilities = event_counts / total_events

    # Calculate Shannon entropy
    entropy = -np.sum(event_probabilities * np.log2(event_probabilities))
    return entropy
# array = (signal)
def get_features(array):
    all_features = np.zeros((1,8))
    all_features[0,0] = feat_mean(array)
    all_features[0,1] = feat_variance(array)
    all_features[0,2] = feat_skewness(array)
    all_features[0,3] = feat_kurtosis(array)
    all_features[0,4] = feat_cov(array)
    all_features[0,5] = feat_mad(array)
    all_features[0,6] = feat_rms(array)
    all_features[0,7] = feat_shannon_entropy(array)
    return all_features


# ## Sliding windows
# the unit of window size and window step is points(second * sample_rate)
# array = (signal)
def slide_windows(array, window_size, window_step):
    array_len = np.size(array)
    num_window = math.floor((array_len-window_size)/window_step)
    output = np.zeros((num_window, window_size))
    for i in range(num_window):
        output[i,:] = array[0 + window_step*i:window_size + window_step*i]
    return output

def channel_slide_windows(array, window_size, window_step): # array's format should be [#channel by #points]
    for i in range(np.size(array,axis=0)):
        if i == 0:
            temp = slide_windows(array = array[0,:], window_size= window_size,
                                window_step = window_step)
            output = np.zeros((np.size(array,axis=0), temp.shape[0], temp.shape[1]))
            output[i,:,:] = temp
        else:
            output[i,:,:] = slide_windows(array = array[i,:], window_size= window_size,
                                    window_step = window_step)
    return output # (#channel, #window, signal)


# ## Normalization_per_sample
def norm_per_sample(data):
    mean_per_sample = np.mean(data,axis=1, keepdims=True)
    std_dev_per_sample = np.std(data,axis=1, keepdims=True)
    
    normalized_data = (data - mean_per_sample) / std_dev_per_sample
    return normalized_data

### Concatnate_npy files
def concatenate_npy_files(folder_path, output_file):
    """
    Concatenate all .npy files in a folder and save the result to an output .npy file.

    Args:
        folder_path (str): Path to the folder containing .npy files.
        output_file (str): Path to the output .npy file where the concatenated data will be saved.
    """
    # Initialize an empty list to store loaded arrays
    arrays = []

    # Loop through the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            loaded_array = np.load(file_path)
            arrays.append(loaded_array)

    # Concatenate the arrays along the desired axis (e.g., axis=0 for vertical stacking)
    # If the arrays have different shapes, make sure they are compatible for concatenation
    result_array = np.concatenate(arrays, axis=0)

    # Save the concatenated array to the output .npy file
    np.save(output_file, result_array)