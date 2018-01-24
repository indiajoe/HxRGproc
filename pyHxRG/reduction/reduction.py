#!/usr/bin/env python 
""" This tool is for recovering the slope/flux image from an HxRG readout data cube """
from __future__ import division
import numpy as np
from scipy.ndimage import filters
import cPickle
from functools32 import lru_cache
from scipy import interpolate
import logging

def subtract_reference_pixels(img,no_channels=32,statfunc=np.median):
    """ Returns the readoud image after subtracting reference pixels of H2RG.
    Input:
         img: 2D full frame Non distructive image readout of HxRG.
         no_channels: Number of channels used in readout. (default:32)
         statfunc: Function(array,axis) which returns the median/mean etc..

         IMP Note- If Pedestal is not subtracted out in img, use statfunc=np.mean
                   If Pedestal is subtracted out, you can use more robust statfunc=np.median
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
        # Correct odd and even columns seperately
        topRefeven = statfunc(channelstrip[:4,0::2])
        topRefodd = statfunc(channelstrip[:4,1::2])  # Calculate median/mean of odd and even columns                    
        botRefeven = statfunc(channelstrip[-4:,0::2])
        botRefodd = statfunc(channelstrip[-4:,1::2])

        Corrected_channelstrip = channelstrip.copy()
        Corrected_channelstrip[:,0::2] = channelstrip[:,0::2] - np.linspace(topRefeven,botRefeven,channelstrip.shape[0])[:,np.newaxis]
        Corrected_channelstrip[:,1::2] = channelstrip[:,1::2] - np.linspace(topRefodd,botRefodd,channelstrip.shape[0])[:,np.newaxis]

        correctedStrips.append(Corrected_channelstrip)

    HRefSubtractedImg = np.hstack(correctedStrips)
    VRef = statfunc(np.hstack((HRefSubtractedImg[:,:4],HRefSubtractedImg[:,-4:])),axis=1)
    return HRefSubtractedImg - VRef[:,np.newaxis]

def fit_slope_zeroIntercept_residue(X,Y):
    """ Returns the residue of a LSQ fitted straight line Y = mX, with intercept fixed to zero """
    X = np.array(X)
    Y = np.array(Y)
    slope = np.sum(Y*X)/np.sum(np.power(X,2))
    return  slope*X - Y 

def fit_slope_1d(X,Y):
    """ Returns the slope and intercept of the the line Y = slope*X +alpha """
    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sxx = np.sum(np.power(X,2))
    Sxy = np.sum(X*Y)
    Syy = np.sum(np.power(Y,2))    
    n = len(X)*1.
    slope = (n*Sxy - Sx*Sy)/(n*Sxx-Sx**2)
    alpha = Sy/n - slope*Sx/n
    return slope, alpha

def fit_slope_andget_residue(X,Y):
    """ Returns the residue of a LSQ fitted straight line Y = mX +c """
    X = np.array(X)
    Y = np.array(Y)
    slope, alpha =  fit_slope_1d(X,Y)
    return  slope*X + alpha - Y 

def robust_medianfromPercentiles(array,percentiles=()):
    """ Estimates the median from percentiles of data in array.
    Warning: Assumes Normal distribution of data in the used percentile ranges
    Useful for estimating median in array when a fraciton of pixels are illuminated 
    Default percentiles used are lower [10,20,30,40,45] percentiles"""

    if percentiles:
        percentiles = np.array(percentiles)
        SigmaVector = scipy.stats.norm.ppf(percentiles/100.)
    else:
        percentiles = np.array([10.,20.,30.,40.,45.])
        SigmaVector = np.array([-1.28155157, -0.84162123, -0.52440051, -0.2533471 , -0.12566135])

    PercentileValues = np.percentile(array,percentiles)
    
    sig, med =  fit_slope_1d(SigmaVector,PercentileValues)

    return med

def subtract_median_bias_residue_channel(ChannelCube,time=None,percentile=50):
    """ Returns the median residue corrected channel strip cube.
    Input:
       ChannelCube: 3D cube data of just one channel strip. 
             Very Important: Cube should be already pedestal subtracted, and Reference pixel bias subtracted.
       time: (optional) epochs of the NDR readouts in the data cube. (default: uniform cadence)
       percentile: 50 is median. Give number between 0 and 100 to choose the percentile of counts to use for slope calculation.
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

    TopBiasResidue = fit_slope_zeroIntercept_residue(time,[np.percentile(tile,percentile) for tile in ChannelCube[:,0:hsize,:]])
    BotBiasResidue = fit_slope_zeroIntercept_residue(time,[np.percentile(tile,percentile) for tile in ChannelCube[:,hsize:,:]])
    
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

           Steps:
             1) Take median of the top and bottom sections of the channel for each readout odd and even seperatly.
             2) Fit the median values with a straight line with zero intercept using LSQ.
                [Intercept and slope is taken to be common for all image: Hence it is crucial to remove pedestal signal before running this function]
                [LSQ fit formula assumes the residues are Gaussian distributed. Hence it is crucial to subtract 0th order 
                 bias values using the refernce pixels at the top and bottom edges of the order before running this correction.]
             3) Interpolate these two median bias level corrections to all the pixels and subtract from the Channel cube (odd and even seperatly).

    """

    TopOddEvenBiases = []
    BottomOddEvenBiases = []

    hsize = int(DataCube.shape[1]/2)
    if time is None:
        time = np.arange(DataCube.shape[0])

    for ChannelCube in np.split(DataCube,np.arange(1,no_channels)*int(2048/no_channels),axis=2):
        # Calculate top bias values (Odd and even seperately concatenated)
        TopOddEvenBiases.append([robust_medianfromPercentiles(tile) for tile in ChannelCube[:,0:hsize,1::2]]+
                                [robust_medianfromPercentiles(tile) for tile in ChannelCube[:,0:hsize,0::2]])
        # Calculate bottom bias values
        BottomOddEvenBiases.append([robust_medianfromPercentiles(tile) for tile in ChannelCube[:,hsize:,1::2]]+
                                   [robust_medianfromPercentiles(tile) for tile in ChannelCube[:,hsize:,0::2]])

    # Fit a straight line and calcuate the residue shifts due to bias fluctuations
    TopResidue = fit_slope_andget_residue(np.tile(time,2*len(TopOddEvenBiases)), np.concatenate(TopOddEvenBiases))
    BottomResidue = fit_slope_andget_residue(np.tile(time,2*len(BottomOddEvenBiases)), np.concatenate(BottomOddEvenBiases))

    TopOddResidues = np.split(TopResidue,2*len(TopOddEvenBiases))[0::2]
    TopEvenResidues = np.split(TopResidue,2*len(TopOddEvenBiases))[1::2]
    BottomOddResidues = np.split(BottomResidue,2*len(BottomOddEvenBiases))[0::2]
    BottomEvenResidues = np.split(BottomResidue,2*len(BottomOddEvenBiases))[1::2]

    # Apply the residue shift correction to odd and even columns of each channel
    CorrectedCubes = []
    for ChannelCube,TOddR,TEvenR,BOddR,BEvenR in zip(np.split(DataCube,np.arange(1,no_channels)*int(2048/no_channels),axis=2),
                                                     TopOddResidues,TopEvenResidues,
                                                     BottomOddResidues,BottomEvenResidues):
        CorrChannelCube = ChannelCube.copy()
        x = np.arange(ChannelCube.shape[1])

        OddResidueCorrectionSlopes = (TOddR - BOddR)/(hsize/2 - (hsize + hsize/2))
        OddResidueCorrection = BOddR[:,np.newaxis] + OddResidueCorrectionSlopes[:,np.newaxis] * (x - (hsize + hsize/2))[np.newaxis,:]
        CorrChannelCube[:,:,1::2] = ChannelCube[:,:,1::2] + OddResidueCorrection[:,:,np.newaxis]

        EvenResidueCorrectionSlopes = (TEvenR - BEvenR)/(hsize/2 - (hsize + hsize/2))
        EvenResidueCorrection = BEvenR[:,np.newaxis] + EvenResidueCorrectionSlopes[:,np.newaxis] * (x - (hsize + hsize/2))[np.newaxis,:]
        CorrChannelCube[:,:,0::2] = ChannelCube[:,:,0::2] + EvenResidueCorrection[:,:,np.newaxis]


        
        CorrectedCubes.append(CorrChannelCube)
    return np.dstack(CorrectedCubes)
        

def remove_biases_in_cube(DataCube,no_channels=32,time=None,do_LSQmedian_correction=-99999):
    """ Returns the data cube after removing variable biases from data cube.
    Input:
        DataCube: The 3D Raw readout Data Cube from an up-the-ramp readout of HxRG.
        no_channels: Number of channels used in readout. (default:32)
        time: (optional) epochs of the NDR readouts in the data cube. (default: uniform cadence)
        do_LSQmedian_correction : (float, default=-99999) Does an extra LSQ based median bias correction, 
                        if the median of the last frame is less then do_LSQmedian_correction value.
                          Hint: Do this only for data which does not saturate in more than 40% of half of each channel.
                          Also only for images where more than 40% of pixels have steady linear flux incedence.
    Output:
       BiasCorrectedDataCube: 3D data cube after correcting all biases.
    
    """
    # Convert to float, just incase it is something else like uint
    if DataCube.dtype not in [np.float, np.float_, np.float16, np.float32, np.float64, np.float128]:
        DataCube = DataCube.astype(np.float)  

    # Step 1: Subtract the pedestal bias levels.
    # Removing these first is important to improve the accuracy of estimates of 
    # various statistics in later steps.
    DataCube -= DataCube[0,:,:]
    
    # Step 2: Estimate bias values from top and bottom reference pixels and subtract them for each channel strip.
    # Step 3: Estimate bias value fluctuation in Vertical direction during the readout time, and subtract them from each strip.
    DataCube = np.array([subtract_reference_pixels(ndr,no_channels=no_channels) for ndr in DataCube])
    
    if do_LSQmedian_correction > robust_medianfromPercentiles(DataCube[-1,:,:]):
        # Step 4: After the previous step the errors in the bias corrections are Gaussian, since it comes 
        # from the error in estimate of bias from small number of reference pixels.
        # So, in the next step we shall fit straight line to the full median image sections of each strip and estimate residual bias corrections.
        DataCube = subtract_median_bias_residue(DataCube,no_channels=no_channels,time=time)

    return DataCube


def remove_bias_preservepedestal_in_cube(DataCube,no_channels=32):
    """ Returns the data cube after removing only variable biases from data cube. And preserves the pedestal bias.
    Input:
        DataCube: The 3D Raw readout Data Cube from an up-the-ramp readout of HxRG.
        no_channels: Number of channels used in readout. (default:32)
    Output:
       VBiasCorrectedDataCube: 3D data cube after removing only the variable biases, preserving pedestal.
    
    """
    # Convert to float, just incase it is something else like uint
    if DataCube.dtype not in [np.float, np.float_, np.float16, np.float32, np.float64, np.float128]:
        DataCube = DataCube.astype(np.float)  

    # Step 1: Estimate bias values from top and bottom reference pixels and subtract them for each channel strip.
    # Step 2: Estimate bias value fluctuation in Vertical direction during the readout time, and subtract them from each strip.
    cleanndrArray = np.array([subtract_reference_pixels(ndr,no_channels=no_channels, statfunc=np.mean) for ndr in DataCube])
    
    # Now calculate mean of all the Bias corrections to obtain pedestal
    Pedestal = np.mean(DataCube - cleanndrArray, axis=0)

    return cleanndrArray + Pedestal


def slope_img_from_cube(DataCube,time):
    """ Fits a slope by linear regression along the axis=0, with respect to time and returns the slope and constant for each pixel as a 2d Matrix
    The linear fitting function is  y = alpha + beta *x
    Equations based on https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
    Parameters:
    -----------
    DataCube   : Masked numpy 3d array.
               Time axis should be axis=0
               The points which shouldn't be used for straight line fitting should be masked out by numpy's ma module.
    time      : 1d numpy array
               The time corresponding to each slice along the time axis of the data. The slope calculated will be in 
               units of this time.
    Returns
    -----------
    (beta,alpha) : masked (2d numpy array, 2d numpy array)
                  The straight line fit is of the equation  y = alpha + beta *x
                  The slope beta and constant alpha is for each pixel is returned as two 2d numpy arrays.
    """
    #The linear regression fit.  y = alpha + beta *x
    #Equation notations based on https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
    #Variables are 2d matrices corresponding to 2d array of pixels.
    tshape = tuple(np.roll(DataCube.shape,-1))  # Creating the tuple to resize the time array to 3d cube. We will have to do a (2,0,1) permutation later. 

    Sx = np.ma.array(np.transpose(np.resize(time,tshape),(2,0,1)),
                     mask=np.ma.getmaskarray(DataCube)).sum(axis=0,dtype=np.float64)
    Sxx = np.ma.array(np.transpose(np.resize(np.square(time),tshape),(2,0,1)),
                      mask=np.ma.getmaskarray(DataCube)).sum(axis=0,dtype=np.float64)

    Sy = DataCube.sum(axis=0,dtype=np.float64)
    # Syy=(np.square(DataCube)).sum(axis=0,dtype=np.float64)  #Only needed to calculate error in slope
    Sxy = (DataCube*time[:,np.newaxis,np.newaxis]).sum(axis=0,dtype=np.float64)
    n = np.ma.count(DataCube,axis=0)   #number of points used in fitting slope of a pixel
    
    beta = (n*Sxy - Sx*Sy)/ (n*Sxx - Sx**2)
    alpha = Sy/n - beta*Sx/n

    #mask beta and alpha where n < 2
    beta = np.ma.masked_where(n<2,beta)
    alpha = np.ma.array(alpha,mask=np.ma.getmaskarray(beta))

    return beta,alpha

def varience_of_slope(slope,NoOfPoints,tframe,redn,gain):
    """ Calculates the varience image of the slope (image) using the formula of Robberto 2010 (Eq 7) when m= 1 case.
    Parameters:
    -----------
    slope: Fitted Slope
    NoOfPoints : Number of points used in slope fitting.
    tframe: Time between each data point (frame time)
    refn: Read noise
    gain: gain of the detector to estimate digitisation noise

    Returns:
    ---------
    Varience : Varience of the slope estimate
    """
    Var = 6*(NoOfPoints**2 + 1)*np.abs(slope) / (5*NoOfPoints*(NoOfPoints**2 -1)*tframe) +\
          12*(redn**2 + gain**2 / 12.)/(NoOfPoints*(NoOfPoints**2 -1)*tframe**2)
    return Var

def piecewise_avgSlopeVar(MaskedDataVector,time,redn,gain):
    """ Returns the average slope and varience by estimating slope at each continous non masked 
    regions in the input MaskedDataVector 
    Parameters:
    -----------
    MaskedDataVector : Masked vector to fit slopes
    time: Time vector for the data points
    redn: Read noise for varience estimate
    gain: gain of the detector to estimate digitisation noise

    Returns:
    ---------
    AvgSlope : Weighted Average of piece wise slopes
    Varience : Varience of the slope estimate
    """    
    localbeta = []
    localn = []
    localvar = []
    #loop over each sections of the ramp.
    slices = np.ma.notmasked_contiguous(MaskedDataVector)
    if slices is None :  #When no unmasked pixels exist
        return np.nan, np.nan

    tf = np.median(np.diff(time)) # The frame time estimate
    for k in range(len(slices)) :
        n = len(MaskedDataVector[slices[k]])
        if  n > 2 : #At least 3 points are there to calculate slope
            t = time[slices[k]]
            Sx = t.sum(dtype=np.float64)
            Sxx = (np.square(t)).sum(dtype=np.float64)
            Sy = MaskedDataVector[slices[k]].sum(dtype=np.float64)
            Sxy = (MaskedDataVector[slices[k]]*t).sum(dtype=np.float64)
            #append localbeta, localalpha, localn and localsigma
            beta = (n*Sxy - Sx*Sy)/ (n*Sxx - Sx**2)
            localbeta.append(beta)
            localn.append(n)
            localvar.append(varience_of_slope(beta,n,tf,redn,gain))
    #calculate the average beta with weights 1/localvarience 
    if len(localvar) > 0 : 
        AvgSlope, weightsum =np.average(localbeta,weights=1.0/np.asarray(localvar),
                                        returned=True)
        Varience = 1/weightsum
        return AvgSlope, Varience
    else :
        return np.nan, np.nan


def apply_nonlinearcorr_polynomial(DataCube,NLcorrCoeff,UpperThresh=None):
    """ Applies the classical non-linearity correction polynomial to Datacube 
    Parameters:
    -----------
    DataCube   : Numpy 3d array.
               Time axis should be axis=0
    NLcorrCoeff: String or numpy 3d array
                Either the file name of the npy file which has the save non linearity correction coefficents.
                Or the 3d cube of coefficients.
    UpperThresh: String or numpy 3d array (optional, default:None)
                Upper Threshold value of the pixel count for each pixel above which non-linearity correction is unreliable and need to be masked.
    Returns
    -----------
    OutDataCube: numpy ma masked 3d cube array
                Outputs is the non linearity corrected 3d Datacube, 
                If UpperThresh is provided it is a numpy masked ma array 
                  with all values which was above UpperThresh masked.
    
    """
    if isinstance(NLcorrCoeff,str):
        NLcorrCoeff = np.load(NLcorrCoeff)

    OutDataCube = np.zeros_like(DataCube)

    for p,coeffs in enumerate(NLcorrCoeff):
        OutDataCube += coeffs[np.newaxis,:,:] * np.power(DataCube,p)

    if UpperThresh is not None:
        if isinstance(UpperThresh,str):
            UpperThresh = np.load(UpperThresh)
        # Mask all data above the threshold
        OutDataCube = np.ma.masked_greater(OutDataCube,UpperThresh)
        
    return OutDataCube

@lru_cache(maxsize=1)
def Load_NonLinCorrBsplineDic(pklfilename):
    """ Loads the pickled Bspline corefficent dictionary into a dictionary of Bsplines """
    NLcorrTCKdic = cPickle.load(open(pklfilename,'rb'))
    BsplineDic = {}
    for (i,j),tck in NLcorrTCKdic.iteritems():
        try:
            BsplineDic[i,j] = interpolate.BSpline(tck[0],tck[1],tck[2],extrapolate=True)
        except TypeError:
            # tck might be None for pixels with no corrections
            BsplineDic[i,j] = None
        except ValueError:
            logging.error("Insufficent coeffs for {0},{1}: {2}".format(i,j,tck))
            BsplineDic[i,j] = None

    return BsplineDic
            

def apply_nonlinearcorr_bspline(DataCube,NLcorrTCKdic,UpperThresh=None, NoOfPreFrames=1):
    """ Applies the non-linearity correction spline model to Datacube 
    Parameters:
    -----------
    DataCube   : Numpy 3d array.
               Time axis should be axis=0  [Warning: DataCube will be overwritten]
    NLcorrTCKdic: String or Dictionary
                Either the file name of the .pkl file which has the dictionary of non-linearity 
                correction Bspline tck coefficents from which to create Bspline dic.
                Or the dictionary of the Bsplines directly
    UpperThresh: String or numpy 2d array (optional, default:None)
                Upper Threshold value of the pixel count for each pixel above which non-linearity 
                correction is unreliable and need to be masked.
    NoOfPreFrames: float or numpy 2d array
               This is the number of frames of flux which got subracted from the DataCube which
               has to be added back before applying non-linearity curve.
               For non-globel reset, this is typically = 1
               For global reset this has to be a 2d array of exposure time in first readout.
               If you set the value to 0, no extra flux will be added.
               Only a rough and robust estimate of flux per pixel is made by taking the 
                      median of differential counts.
    Returns
    -----------
    OutDataCube: numpy ma masked 3d cube array
                Outputs is the non linearity corrected 3d Datacube, 
                If UpperThresh is provided it is a numpy masked ma array 
                  with all values which was above UpperThresh masked.
    
    """
    if isinstance(NLcorrTCKdic,str):
        NLcorrTCKdic = Load_NonLinCorrBsplineDic(NLcorrTCKdic)

    if NoOfPreFrames: # If NoOfPreFrames is not Zero, we have to caluclate flux and add
        CrudeFlux = np.median(np.diff(DataCube,axis=0),axis=0)
        # Add the Flux to DataCube
        DataCube += CrudeFlux*NoOfPreFrames

    OutDataCube = DataCube  # Overwrite the same array to save memory

    # Do the Non-linearity correction
    for (i,j),bspl in NLcorrTCKdic.iteritems():
        try:
            OutDataCube[:,i,j] = bspl(DataCube[:,i,j])
        except TypeError:
            # Bspline is None for pixels with no corrections
            OutDataCube[:,i,j] = DataCube[:,i,j]

    if UpperThresh is not None:
        if isinstance(UpperThresh,str):
            UpperThresh = np.load(UpperThresh)
        # Mask all data above the threshold
        OutDataCube = np.ma.masked_greater(OutDataCube,UpperThresh)
        
    return OutDataCube


def abrupt_change_locations(DataCube,thresh=20):
    """ Returns the array of positions at which abrupt change occured due to reset or Cosmic Ray hits.
    Uses the [1,-3,3,-1] digital filter find abrupt changes.
    Parameters:
    -----------
    DataCube   : Numpy 3d array.
               Time axis should be axis=0
    thresh (default: 20):
               Threshold to detect the change in filter convolved image
    Returns
    -----------
    (T,I,J) : The locations of abrupt changes like CR hit in the tuple of 3 arrays format (Time,I,J).
    """
    # Do a 3 pixels median filtering in the data to reduce the read noise in time series and preserve steps
    if np.ma.isMaskedArray(DataCube) :
        MFDataCube = filters.median_filter(DataCube.data,size=(3,1,1),mode='nearest')
    else:
        MFDataCube = filters.median_filter(DataCube,size=(3,1,1),mode='nearest')

    convimg = MFDataCube[:-3,:,:] -3*MFDataCube[1:-2,:,:] +3*MFDataCube[2:-1,:,:] -MFDataCube[3:,:,:] #Convolution of image with [1,-3,3,-1] along time axis
        
    medianconv = np.median(convimg,axis=0)  #Wrongly nameing the variable now itself to conserve memory
    stdconv = np.median(np.abs(convimg-medianconv),axis=0) * 1.4826  # Convert MAD to std  # MAD for robustnus
    #We should remove all the points where number of data points were less than 5
    stdconv[np.where(np.ma.count(DataCube,axis=0) < 5)] = 99999
    # We should also remove any spuriously small stdeviation, say those lees than median std deviation  
    MedianNoise = np.median(stdconv)
    stdconv[np.where(stdconv < MedianNoise)] = MedianNoise  #This will prevent spuriously small std dev estimates
    #Find the edges of ramp jumps.
    if np.ma.isMaskedArray(DataCube) :
        (T,I,J) = np.ma.where(np.ma.array(convimg-medianconv,mask=np.ma.getmaskarray(DataCube[3:,:,:])) > thresh*stdconv) 
    else: 
        (T,I,J) = np.ma.where(convimg-medianconv > thresh*stdconv)
    T=T+2  #Converting to the original time coordinate of DataCube by correcting for the shifts. This T is the first pixel after CR hit
    return T,I,J
