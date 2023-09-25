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
_____
Data 9/24/2023 <br>
* I separated Paper_parallel_memristive_convolution_network into 'FeatureSelection&ArraySave' and 'Network_Train_Evaluation'. 
  * FeatureSelection: Calculating 8 features, using PCA reduce feature dimension into 64, Save (# sample, # feature) arrays. 
  * Network_Train_Evaluation: Applying saved metrix to train model created by paper's author and generate evaluation metrix. 
_____



# Reference
* Siddiqui, M. K., Morales-Menendez, R., Huang, X., & Hussain, N. (2020). A review of epileptic seizure detection using machine learning classifiers. Brain informatics, 7(1), 1-18.
* Li, C., Lammie, C., Amirsoleimani, A., Azghadi, M. R., & Genov, R. (2023). Simulation of memristive crossbar arrays for seizure detection and prediction using parallel Convolutional Neural Networks. Software Impacts, 15, 100473.

# Dependency
* numpy
* statistics
* pyedflib
* sklearn
* matplotlib
* IPython
