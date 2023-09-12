# Seizure-Detection
Date: 9/11/2023
## Dataset Introduction
Source: CHB-MIT Scalp EEG Database; Link: https://physionet.org/content/chbmit/1.0.0/
<br>
## Code Introduction
**seizure_detection_phase_1**: 
* Read data(chb07_13)
* PCA method to reduce raw signal dimensionality
* Features calculation: Mean, variance, Mode, Median, Peaks, Signal energy(Time Domain), Approximate entropy(Not results generated), Spectrum energy, Continuous wavelet transform(CWT)

**Paper_parallel_memristive_convolution_network**:<br>
This code is based on the paper called Seizure_Detection_and_Prediction_by_Parallel_Memristive_Convolutional_Neural_Networks and author: Chenqi Li
* Feature selection: Mean, Variance, Skewness, Kurtosis, Coefficient of Variation, Median absolute deviation, Root mean square amplitude, Shannon entropy. 
<br>Continue working in this code...

# Reference
* A review of epileptic seizure detection using machine learning classifiers, Mohammad Khubeb Siddiqui
* Seizure_Detection_and_Prediction_by_Parallel_Memristive_Convolutional_Neural_Networks
