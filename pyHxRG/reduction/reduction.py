#!/usr/bin/env python 
""" This tool is for recovering the slope/flux image from an HxRG readout data cube """
from __future__ import division

def subtract_reference_pixels(img,no_channels=32):
    """ Returns the readoud image after subtracting reference pixels of H2RG.
    Input:
         img: 2D full frame Non distructive image readout of HxRG.
         no_channels: Number of channels used in readout. (default:32)
    Output:
         2D image after subtracting the Reference pixel biases using the following procedure.
         Steps:
           1) For each channel, calculate the mean of the median counts in odd and even pixels of the top and bottom reference pixels.
           2) Linearly interpolate this top and bottom reference values across the coulmn strip and subtract.
           3) Combine each channel strips back to a single image.
           4) Median combine horizontally the vertical 4 coulmns of Refernece pixels on both edges of the array.
           5) Subtract this single column bias drift from all the columns in the array.
"""
    correctedStrips = []
    for channelstrip in np.split(img,np.arange(1,no_channels)*int(2048/no_channels),axis=1):
        topRef = np.mean([np.median(channelstrip[:4,0::2]),np.median(channelstrip[:4,1::2])])  # Calculate mean of median of odd and even columns                    
        botRef = np.mean([np.median(channelstrip[-4:,0::2]),np.median(channelstrip[-4:,1::2])])
        correctedStrips.append(channelstrip - np.linspace(topRef,botRef,channelstrip.shape[0])[:,np.newaxis])

    HRefSubtractedImg = np.hstack(correctedStrips)
    VRef = np.median(np.hstack((HRefSubtractedImg[:,:4],HRefSubtractedImg[:,-4:])),axis=1)
    return HRefSubtractedImg - VRef[:,np.newaxis]

def fit_slope_zeroIntercept_residue(X,Y):
    """ Returns the residue of a LSQ fitted straight line Y = mX, with intercept fixed to zero """
    X = np.array(X)
    Y = np.array(Y)
    slope = np.sum(Y*X)/np.sum(np.power(X,2))
    return  slope*X - Y 

def subtract_median_bias_residue_channel(ChannelCube,time=None):
    """ Returns the median residue corrected channel strip cube.
    Input:
       ChannelCube: 3D cube data of just one channel strip. 
             Very Important: Cube should be already pedestal subtracted, and Reference pixel bias subtracted.
       time: (optional) epochs of the NDR readouts in the data cube. (default: uniform cadence)
    Output:
       CorrectedChannelCube: 3D cube of residue corrected channels.
           The residue is calculated by the following technique.
           Steps:
             1) Take median of the top and bottom sections of the channel for each readout.
             2) Fit the median values with a straight line with zero intercept using LSQ.
                [Intercept is fixed to Zero: Hence it is crucial to remove pedestal signal before running this function]
                [LSQ fit formula assumes the residues are Gaussian distributed. Hence it is crucial to subtract 0th order 
                 bias values using the refernce pixels at the top and bottom edges of the order before running this correction.]
             3) Interpolate these two median bias level corrections to all the pixels and subtract from the Channel cube.
    """

    if time is None:
        time = np.arange(ChannelCube.shape[0])
    hsize = int(ChannelCube.shape[1]/2)

    TopBiasResidue = fit_slope_zeroIntercept_residue(time,[np.median(tile) for tile in ChannelCube[:,0:hsize,:]])
    BotBiasResidue = fit_slope_zeroIntercept_residue(time,[np.median(tile) for tile in ChannelCube[:,hsize:,:]])
    
    ResidueCorrectionSlopes = (TopBiasResidue - BotBiasResidue)/(hsize/2 - (hsize + hsize/2))
    x = np.arange(ChannelCube.shape[1])
    ResidueCorrection = BotBiasResidue[:,np.newaxis] + ResidueCorrectionSlopes[:,np.newaxis] * (x - (hsize + hsize/2))[np.newaxis,:]
    return ChannelCube + ResidueCorrection[:,:,np.newaxis]

def subtract_median_bias_residue(DataCube,no_channels=32,time=None):
    """ Returns the median residue bias corrected data cube.
    Input: 
        DataCube: The 3D pedestal subtracted, and reference pixel subracted Data Cube. 
        no_channels: Number of channels used in readout. (default:32)
        time: (optional) epochs of the NDR readouts in the data cube. (default: uniform cadence)
    Output:
        BiasCorrectionCorrectedCube : 3D cube after correcting the bias corrections across channel
    """
    CorrectedCubes = []
    for ChannelCube in np.split(DataCube,np.arange(1,no_channels)*int(2048/no_channels),axis=2):
        CorrectedCubes.append( subtract_median_bias_residue_channel(ChannelCube,time=time) )
    return np.dstack(CorrectedCubes)
        

def remove_biases_in_cube(DataCube,no_channels=32,time=None,do_LSQmedian_correction=True):
    """ Returns the data cube after removing variable biases from data cube.
    Input:
        DataCube: The 3D Raw readout Data Cube from an up-the-ramp readout of HxRG.
        no_channels: Number of channels used in readout. (default:32)
        time: (optional) epochs of the NDR readouts in the data cube. (default: uniform cadence)
        do_LSQmedian_correction : (bool, default=True) Does an extra LSQ based median bias correction.
                          Do this only for data which does not saturate in more than 40% of half of each channel.
                          Also only for images where more than 40% of pixels have steady linear flux incedence.
    Output:
       BiasCorrectedDataCube: 3D data cube after correcting all biases.
    
    """
    # Step 1: Subtract the pedestal bias levels.
    # Removing these first is important to improve the accuracy of estimates of 
    # various statistics in later steps.
    DataCube = DataCube.astype(np.float)  # Convert to float, just incase it is something else like uint
    DataCube = DataCube - DataCube[0,:,:]
    
    # Step 2: Estimate bias values from top and bottom reference pixels and subtract them for each channel strip.
    # Step 3: Estimate bias value fluctuation in Vertical direction during the readout time, and subtract them from each strip.
    DataCube = np.array([subtract_reference_pixels(ndr,no_channels=no_channels) for ndr in DataCube])
    
    if do_LSQmedian_correction:
        # Step 4: After the previous step the errors in the bias corrections are Gaussian, since it comes 
        # from the error in estimate of bias from small number of reference pixels.
        # So, in the next step we shall fit straight line to the full median image sections of each strip and estimate residual bias corrections.
        DataCube = subtract_median_bias_residue(DataCube,no_channels=no_channels,time=time)

    return DataCube

