""" This is a static parameter file containing default detector 
    parameters and characteristics """

############# Detector Noise Parameters #################
# By default Use parameters that generate noise similar to JWST NIRSpec
#
# Ref: Code provided by http://adsabs.harvard.edu/abs/2015PASP..127.1144R
######
mknoise_kargs = { 'rd_noise' : 4.0,  # White read noise per integration
                  'pedestal' : 4.0,  # DC pedestal drift rms
                  'c_pink' : 3.0,  # Correlated pink noise
                  'u_pink' : 1.0, # Uncorrelated pink noise
                  'acn' : 0.5, # Correlated ACN
                  'pca0_amp' : 0.2   # Amplitude of PCA zero "picture frame" noise
              }
######

######## Dark current rate ###########
dark_current = 0.005  # e-/s/pix

######## Inter-pixel capacitance (IPC) #########
# Provide the Normalise convolution matrix (as numpy array) for simulating IPC
# Enter None for skipping this step
import numpy as np
ipc_matrix = np.array([[0,    0.004,    0],
                       [0.008,0.975,0.009],
                       [0,    0.004,    0]])
######

#################### ng.HXRGNoise input arguments ##############
#### See the doc string of ng.HXRGNoise  for details of all the allowed list of input parameters
HXRGNoise_kargs = {'naxis1':2048,
                   'naxis2':2048, 
                   'n_out':4,   #  Number of detector outputs
                   'dt': 1.e-5, # Pixel dwell time in seconds
                   'nroh': 12, # nroh and nfoh are in units of clock steps
                   'nfoh' : 1,
                   'verbose' : False}  

############################ Output gain
##### e-/ADU gain factor for converting final output to ADU
############## This should include both HxRG and ASIC or Leach Box gains
epadu = 2.5

