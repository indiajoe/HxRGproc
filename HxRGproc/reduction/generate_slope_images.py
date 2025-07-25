#!/usr/bin/env python
""" This script is to create slope images from the Up the ramp images"""
from __future__ import division
import sys
import argparse
from astropy.io import fits
import numpy as np
import numpy.ma
import os
import re
import errno
from datetime import datetime
import astropy.units as u
from multiprocessing import TimeoutError
from multiprocessing.pool import Pool
import logging
import signal
import traceback
from astropy.stats import biweight_location
from . import reduction 
from .instruments import SupportedReadOutSoftware_for_slope as READOUT_SOFTWARE

try:
    from ConfigParser import SafeConfigParser
    from functools32 import wraps, partial
except ModuleNotFoundError:  # Python 3 environment
    from configparser import ConfigParser as SafeConfigParser
    from functools import wraps, partial



def pack_traceback_to_errormsg(func):
    """Decorator which packes any raised error traceback to its msg 
    This is useful to pack a child process function call while using multiprocess """
    @wraps(func)
    def wrappedFunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\nOriginal {}".format(e, traceback.format_exc())
            raise type(e)(msg)
    return wrappedFunc

def log_all_uncaught_exceptions_handler(exp_type, exp_value, exp_traceback):
    """ This handler is to override sys.excepthook to log uncaught exceptions """
    logging.error("Uncaught exception", exc_info=(exp_type, exp_value, exp_traceback))
    # call the original sys.excepthook
    sys.__excepthook__(exp_type, exp_value, exp_traceback)


def log_memory_errors(func):
    """ This is a decorator to log Memory errors """
    @wraps(func)
    def wrappedFunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError as e:
            logging.critical('Memory Error: Please free up Memory on this computer to continue..')
            logging.exception('Memory Error while calling {0}'.format(func.__name__))
            logging.info('Hint: Reduce number of processes running in parallel on this computer')
            raise
    return wrappedFunc

def load_data_cube(filelist):
    """ Loads the input file list in to a data cube, and return both datacube and fits header seperately.
    Input: 
         filelist : List of sorted filenames of the up-the-ramp data
    Returns:
         DataCube, fitsheader of first img
    """
    logging.info('Loading UTR: {0}'.format(filelist))
    # datalist = [subtractReferencePixels(fits.getdata(f)) for f in sorted(filelist)]
    datalist = [fits.getdata(f) for f in filelist]
    return np.array(datalist), fits.getheader(filelist[0])


def calculate_slope_image(UTRlist,Config,NoOfFSkip=0):
    """ Returns slope image hdulist object, and header from the input up-the-ramp fits file list 
    Input:
          UTRlist: List of up-the-ramp NDR files
          Config: Configuration dictionary imported from the Config file
          NoOfFSkip: Number of frames skipped after Reset
    Returns:
         Slopeimage hdulist object, header
"""
    HDR_INTTIME = READOUT_SOFTWARE[Config['ReadoutSoftware']]['HDR_INTTIME']
    HDR_NOUTPUTS = READOUT_SOFTWARE[Config['ReadoutSoftware']]['HDR_NOUTPUTS']
    # Fix any fixes needed for raw header as well as any artifacts in DataCube
    FixHeader_func = READOUT_SOFTWARE[Config['ReadoutSoftware']]['FixHeader_func']
    FixDataCube_func = READOUT_SOFTWARE[Config['ReadoutSoftware']]['FixDataCube_func']
    DataCube, header = load_data_cube(UTRlist) 
    header = FixHeader_func(header,fname=UTRlist[-1])
    DataCube = FixDataCube_func(DataCube)
    time = np.array([FixHeader_func(fits.getheader(f),fname=f)[HDR_INTTIME] for f in UTRlist])

    header['PEDESVAL'] = biweight_location(DataCube[0])  # Save the pedestal value in header

    pedestal_image = np.copy(DataCube[0])

    NoNDR = DataCube.shape[0]
    logging.info('Number of NDRs = {0}'.format(NoNDR))
    # Bias level corrections
    if Config['DoPedestalSubtraction']: 
        logging.info('Subtracted Pedestal')
        DataCube = reduction.remove_biases_in_cube(DataCube,time=time,
                                                   no_channels=header[HDR_NOUTPUTS],
                                                   do_LSQmedian_correction=Config['DoLSQmedianCorrection'],
                                                   vertical_smooth_window=Config['VerticalReferenceSmoothWindow'])
    else:
        DataCube = reduction.remove_bias_preserve_pedestal_in_cube(DataCube,
                                                                   no_channels=header[HDR_NOUTPUTS],
                                                                   vertical_smooth_window=Config['VerticalReferenceSmoothWindow'])

    # Non-linearity Correction
    if Config['NonLinearCorrCoeff']:
        logging.info('Applying NonLinearity Corr:{0}'.format(Config['NonLinearCorrCoeff']))
        if os.path.basename(Config['NonLinearCorrCoeff'])[0:4] == 'POLY':
            # File name wiht POLY* is a polynomical correction
            DataCube = reduction.apply_nonlinearcorr_polynomial(DataCube,
                                                                Config['NonLinearCorrCoeff'],
                                                                UpperThresh=None)
        elif os.path.basename(Config['NonLinearCorrCoeff'])[0:4] == 'BSPL':
            # File name with BSPL* is a BSpline correction
            DataCube = reduction.apply_nonlinearcorr_bspline(DataCube,
                                                             Config['NonLinearCorrCoeff'],
                                                             UpperThresh=None, NoOfPreFrames=NoOfFSkip+1)

    # Mask values above upper threshold before fitting slope
    if (Config['UpperThreshold'] is not None) or (Config['ADCThreshold'] is not None):
        if isinstance(Config['UpperThreshold'],str):
            UpperThresh = np.load(Config['UpperThreshold'])  
        elif isinstance(Config['UpperThreshold'],(int,float)):
            UpperThresh = Config['UpperThreshold']
        else:
            logging.info('No Upper thresholding.')
            UpperThresh = None

        if isinstance(Config['ADCThreshold'],(int,float)):
            if Config['DoPedestalSubtraction']: 
                ADCThresh = Config['ADCThreshold'] - pedestal_image
            else:
                ADCThresh = Config['ADCThreshold']
        else:
            logging.info('No ADC thresholding.')
            ADCThresh = None

        # Combine the two threshold criteria into one
        if ADCThresh is not None:
            if UpperThresh is not None:
                if (np.isscalar(UpperThresh) and np.isscalar(ADCThresh)) or ( (not np.isscalar(UpperThresh)) and  (not np.isscalar(ADCThresh))):
                    UpperThresh = np.min([UpperThresh,ADCThresh],axis=0)
                elif np.isscalar(ADCThresh):
                    UpperThresh[UpperThresh>ADCThresh] = ADCThresh
                elif np.isscalar(UpperThresh):
                    ADCThresh[ADCThresh>UpperThresh] = UpperThresh
            else:
                UpperThresh = ADCThresh  # ADCThresh is the only threshold

        DataCube = np.ma.masked_greater(DataCube,UpperThresh)

        # Number of NDRs used in slope fitting after simple threshold.
        NoNDRArray = np.ma.count(DataCube,axis=0)

        # If user has asked to make exposure constant in certain regions
        if Config['ConstantExposureRegion']:
            if isinstance(Config['ConstantExposureRegion'],str):
                CExpRegionArray = np.load(Config['ConstantExposureRegion'])  
            else:
                # Use all array as same region
                CExpRegionArray = np.ones((DataCube.shape[1],DataCube.shape[2]))
            ExpCutoffPercentile = Config['CER_CutoffPercentile']
            for region in np.unique(CExpRegionArray):
                if region == 0 : 
                    continue # Zero is for regions where no uniformity needs to be done
                RegionMask = CExpRegionArray == region
                # Find the cutoff NDR number
                CutNDR = np.percentile(NoNDRArray[RegionMask],ExpCutoffPercentile,
                                       interpolation='lower')
                DataCube[CutNDR:,RegionMask] = np.ma.masked

    # Mask initial readout of pixels which showed spurious values in first readout
    # Use the [1,-3,3,-1] digital filter to detect abrupt changes in the up-the-ramp
    T,I,J = reduction.abrupt_change_locations(DataCube,thresh=20)
    CR_TIJ = [] # Cosmicray events list
    ResetAnomalyPixels = set()
    for t,i,j in zip(T,I,J):
        if t <=2:
            DataCube[:2,i,j] = np.ma.masked
            ResetAnomalyPixels.add((i,j))
        else:
            CR_TIJ.append((t,i,j))  # save pure CR events

    # Number of NDRs actually used in slope fitting.
    NoNDRArray = np.ma.count(DataCube,axis=0)

    # Convert images from ADU to electrons
    gain = Config['GainEPADU']
    DataCube *= gain
    logging.info('Fitting slope..')
    slopeimg,alpha = reduction.slope_img_from_cube(DataCube, time)

    # convert all masked values to nan
    slopeimg = np.ma.filled(slopeimg,fill_value=np.nan)

    redn = Config['ReadNoise']* gain

    if Config['CalculateVarienceImage']:
        tf = np.median(np.diff(time)) # The frame time estimate
        VarImg = reduction.varience_of_slope(slopeimg,NoNDRArray,tf,redn,gain)

    TotalCRhits = len(CR_TIJ)
    if TotalCRhits < Config['MaxNoOfCRfix']:
        # ReCalculate correct slope for Cosmic ray hit points
        logging.info('Fixing {0} CR hit slopes..'.format(TotalCRhits))
        if TotalCRhits > 0:  # Mask only if the CR_TIJ is not an empty list
            DataCube[tuple(zip(*CR_TIJ))] = np.ma.masked  # Mask all points just after CR hit
            for t,i,j in CR_TIJ:
                slopeimg[i,j], var = reduction.piecewise_avg_slope_var(DataCube[:,i,j],time,redn,gain)
                if Config['CalculateVarienceImage']:
                    VarImg[i,j] = var
        header['history'] = 'Cosmic Ray hits fixed before slope calculation'
    else:
        logging.info('UnFixed {0} CR hits..'.format(TotalCRhits))

    header['NoNDR'] = (NoNDR, 'No of NDRs used in slope')
    header['EXPLNDR'] = (time[-1], 'Int Time of Last NDR used in slope')
    header['ITIME'] = time[-1]  # Update the ITIME in the header of the first image with the last NDR's
    header['PEDSUB'] = (Config['DoPedestalSubtraction'], 'T/F Did Pedestal Subtraction')
    header['MLSQBIA'] = (Config['DoLSQmedianCorrection'], 'UpperThreshold to do LSQ median bias algo')
    header['VREFSMO'] = (Config['VerticalReferenceSmoothWindow'], 'Vertical Reference Smoothing Window')
    header['NLCORR'] = (os.path.basename(str(Config['NonLinearCorrCoeff'])), 'NonLinearCorr Coeff File')
    header['UTHRESH'] = (os.path.basename(str(Config['UpperThreshold'])), 'UpperThreshold Mask value/file')
    header['CUNITS'] = ('e-/sec','Units of the counts in image')
    header['EPADU'] = (gain,'Gain e/ADU')
    header['READNOS'] = (redn,'Single NDR Read Noise (e- rms)')
    header['NOOFCRH'] = (TotalCRhits,'No of Cosmic Rays Hits detected')
    header['NRESETA'] = (len(ResetAnomalyPixels),'No of Reset Anomaly pixels detected')
    header['history'] = 'Slope image generated'
    hdu = fits.PrimaryHDU(slopeimg.astype('float32'),header=header)
    hdulist = fits.HDUList([hdu])

    if Config['CalculateVarienceImage']:
        # Save the varience image as a fits extention
        hduVar = fits.ImageHDU(VarImg.astype('float32'))
        hduVar.header['CUNITS'] = ('(e-/sec)^2','Units of the counts in image')
        hduVar.header['COMMENT'] = 'Varience Image of the Slope image'
        # Create multi extension fits file
        hdulist.append(hduVar)
        
    if Config['UpperThreshold']:
        # Save the No# of NDRs used in each pixel also as a fits extension
        hduNoNDRs = fits.ImageHDU(NoNDRArray.astype('int16'))
        hduNoNDRs.header['COMMENT'] = 'No# of NDRS used in Slope fitting'
        # Create multi extension fits file
        hdulist.append(hduNoNDRs)

    # If user has asked to make exposure up-the-ramp curve for dignositices
    if Config['AverageUpTheRampDiagnostic']:
        if isinstance(Config['AverageUpTheRampDiagnostic'],str):
            RegionFilename = Config['AverageUpTheRampDiagnostic']
            MultiRegionArray = np.load(RegionFilename)  
        else:
            # Use all array as same region
            MultiRegionArray = np.ones((DataCube.shape[1],DataCube.shape[2]))
            RegionFilename = 'ALL'
        AverageRamps = []
        for region in np.unique(MultiRegionArray):
            if region == 0 : 
                continue # Zero is for regions to ignore
            RegionMask = MultiRegionArray == region
            AverageRamps.append(np.nanmean(DataCube[:,RegionMask].filled(np.nan),axis=1))

        # Save the Average Ramp curves of each region also as a fits extension
        hduAvgRamps = fits.ImageHDU(np.array(AverageRamps).astype('float32'))
        hduAvgRamps.header['REGFILE'] = (os.path.basename(RegionFilename),'Filename of RegionMask/ALL')
        hduAvgRamps.header['DELTAT'] = (np.median(np.diff(time)), 'Time delta between readout in sec')
        SlopeDerivative = np.diff(np.nanmean(AverageRamps,axis=0))
        try:
            hduAvgRamps.header['MINMAXD'] = (np.nanmax(SlopeDerivative) - np.nanmin(SlopeDerivative), 'Max - Min of the derivative of average slope')
            hduAvgRamps.header['STD_D'] = (np.nanstd(SlopeDerivative), 'Std dev of the derivative of average slope')
        except ValueError as e:
            logging.error(e)
            logging.error('Unable to calculate minmax or dtd due to {0} nans'.format(len(SlopeDerivative)))
            hduAvgRamps.header['MINMAXD'] = (0, 'Max - Min of the derivative of average slope')
            hduAvgRamps.header['STD_D'] = (0, 'Std dev of the derivative of average slope')

        hduAvgRamps.header['COMMENT'] = 'Average up-the-ramp curves of each region'
        # Create multi extension fits file
        hdulist.append(hduAvgRamps)


    return hdulist,header

# @pack_traceback_to_errormsg can be commented out when function not used in multiprocess
@pack_traceback_to_errormsg
@log_memory_errors
def generate_slope_image(RampNo,InputDir,OutputDir, Config, OutputFileFormat='Slope-R{0}.fits',
                         FirstNDR = 0, LastNDR = None,
                         RampFilenameString='H2RG_R{0:02}_M',
                         FilenameSortKeyFunc = None,
                         ExtraHeaderDictFunc= None):
    """ 
    Generates the Slope image and writes the output slope image for the data files.
    Input:
        RampNo: (int) Ramp number (Ex: 1)
        InputDir: (str) Input Directory containing the UTR files
        OutputDir: (str) Output Directory to write output slope images into
        Config: Configuration dictionary imported from the Config file
        OutputFileFormat: (str,optional) Output file format (default: 'Slope-R{0}.fits')
        FirstNDR = (int,optional) Number of initial NDRs to skip for slope fitting (default: 0) 
        LastNDR = (int/None,optional) Number of the maximum NDR to use in slope fitting (default: None means all)
        RampFilenameString= (str,optional) Filename substring format which uniquely identifies a particular Ramp file (Default: 'H2RG_R{0:02}_M')
        FilenameSortKeyFunc = (func) Function(filename) which returns the key to use for sorting filenames
        ExtraHeaderDictFunc= (func,None optinal) Function(header) which returns a dictionary of any extra header keywords 
                             to add. (Default : None)
    
    Returns:
        OutputFileName : Filename of the output slope fits file which was written
    """
    if FilenameSortKeyFunc is None:
        # Use the file name string itself to sort!
        logging.warning('No function to sort filename provided, using default filename sorting..')
        FilenameSortKeyFunc = lambda f: f

    OutputFileName = os.path.join(OutputDir,OutputFileFormat.format(RampNo))
    try:
        os.makedirs(OutputDir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            logging.error(e)
            raise
        pass # Ignore the error that Output dir already exist.

    if os.path.isfile(OutputFileName):
        logging.warning('WARNING: Output file {0} already exist...'.format(OutputFileName))
        logging.warning('Skipping regeneration of the slope image')
        return None
        
    # First search for all raw files
    RampidRegexp = READOUT_SOFTWARE[Config['ReadoutSoftware']]['RampidRegexp']  # We will use the RampidRegexp to filter out any bad fits filenames which doesnot belong
    UTRlistT = sorted((os.path.join(InputDir,f) for f in os.listdir(InputDir) if ((os.path.splitext(f)[-1] == '.fits') and \
                                                                                  (RampFilenameString.format(RampNo) in os.path.basename(f)) and \
                                                                                  (re.search(RampidRegexp,os.path.basename(f)) is not None))))
    UTRlist = sorted(UTRlistT,key=FilenameSortKeyFunc)

    if LastNDR is None:
        LastNDR = len(UTRlist)

    logging.info('Processing Ramp data of {0}'.format(RampNo))
 
    sanitycheckUTR = (LastNDR <= len(UTRlist)) and (FirstNDR < len(UTRlist)) # Sanity check of the user input of range of NDRs

    if (len(UTRlist[FirstNDR:LastNDR]) > 1) and sanitycheckUTR: # You need atlest two frames to fit slope
        Slopehdulist,header = calculate_slope_image(UTRlist[FirstNDR:LastNDR], Config, NoOfFSkip=FirstNDR)
    else:
        logging.warning('Insuffient number of NDRs (={0}) to generate slope image'.format(len(UTRlist[FirstNDR:LastNDR])))
        logging.warning('Sanity Check of First:LastNDR = {0}:{1} -{2}'.format(FirstNDR,LastNDR,sanitycheckUTR))
        logging.warning('Skipping image {0}: {1}'.format(RampNo,UTRlist[FirstNDR:LastNDR]))
        return None

    if ExtraHeaderDictFunc is not None:
        for key,value in ExtraHeaderDictFunc(header).items():
            Slopehdulist[0].header[key] = value

    if not os.path.exists(OutputDir): # Verify once again the output directory exist
        os.makedirs(OutputDir)

    Slopehdulist.writeto(OutputFileName)

    return OutputFileName



def parse_str_to_types(string):
    """ Converts string to different object types they represent.
    Supported formats: True,Flase,None,int,float,list,tuple"""
    if string == 'True':
        return True
    elif string == 'False':
        return False
    elif string == 'None':
        return None
    elif string == '""':
        return ""
    elif string.lstrip('-+ ').isdigit():
        return int(string)
    elif (string[0] in '[(') and (string[-1] in ')]'): # Recursively parse a list/tuple into a list
        return [parse_str_to_types(s) for s in string.strip('()[]').split(',')]
    else:
        try:
            return float(string)
        except ValueError:
            return string
        
        

def create_configdict_from_file(configfilename):
    """ Returns a configuration object by loading the config file """
    Configloader = SafeConfigParser()
    Configloader.optionxform = str  # preserve the Case sensitivity of keys
    Configloader.read(configfilename)
    # Create a Config Dictionary
    Config = {}
    for key,value in Configloader.items('slope_settings'):
        Config[key] = parse_str_to_types(value)
    for key,value in Configloader.items('filename_settings'):
        Config[key] = parse_str_to_types(value)
    return Config

def parse_args():
    """ Parses the command line input arguments for HxRG data reduction"""
    parser = argparse.ArgumentParser(description="Script to Generate Slope/Flux images from Up-the-Ramp data taken using Teledyne's HxRG detector")
    parser.add_argument('InputDir', type=str,
                        help="Input Directory contiaining the Up-the-Ramp Raw data files. Multiple directories can be provided comma seperated.")
    parser.add_argument('OutputMasterDir', type=str,
                        help="Output Master Directory to which output images to be written")
    parser.add_argument('ConfigFile', type=str,
                        help="Configuration File which contains settings for Slope calculation")
    parser.add_argument('--NoNDR_Drop_G', type=str, default=None,
                        help="No of NDRS per Group:No of Drops:No of Groups (Example  40:60:5)")
    parser.add_argument('--FirstNDR', type=int, default=0,
                        help="Number of First NDRs to be skipped")
    parser.add_argument('--LastNDR', type=int, default=None,
                        help="Maximum NDRs to be used. (Default: all)")
    parser.add_argument('--noCPUs', type=int, default=1,
                        help="Number of parallel CPUs to be used to process independent Ramps in parallel")
    parser.add_argument('--logfile', type=str, default=None,
                        help="Log Filename to write logs during the run")
    parser.add_argument("--loglevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help="Set the logging level")

    args = parser.parse_args()
    return args
    
def main():
    """ Standalone Script to generate Slope images from Up the Ramp data taken using Teledyne's HxRG detector"""
    # Override the default exception hook with our custom handler
    sys.excepthook = log_all_uncaught_exceptions_handler

    args = parse_args()    

    if args.logfile is None:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                            level=logging.getLevelName(args.loglevel))
    else:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                            level=logging.getLevelName(args.loglevel), 
                            filename=args.logfile, filemode='a')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # Sent info to the stdout as well

    InputDirList = args.InputDir.split(',')

    for InputDir in InputDirList: 
        logging.info('Processing data in {0}'.format(InputDir))

        Config = create_configdict_from_file(args.ConfigFile)
        logging.info('Slope Configuration: {0}'.format(Config))
        OutputDir = os.path.join(args.OutputMasterDir,os.path.basename(InputDir.rstrip('/')))

        InputDir = os.path.join(InputDir,Config['InputSubDir']) # Append any redundant input subdirectory to be added
        if not os.path.isdir(InputDir):
            logging.error('No image folder {0}'.format(InputDir))
            logging.info('No images to process in {0}'.format(InputDir))
            continue # skip to next directory

        # Find the number of Ramps in the input Directory
        RampidRegexp = READOUT_SOFTWARE[Config['ReadoutSoftware']]['RampidRegexp']
        imagelist = sorted((os.path.join(InputDir,f) for f in os.listdir(InputDir) if ((os.path.splitext(f)[-1] == '.fits') and (re.search(RampidRegexp,os.path.basename(f)) is not None))))
        RampList = sorted(set((re.search(RampidRegexp,os.path.basename(f)).group(1) for f in imagelist))) # Ex: 45 in H2RG_R45_M01_N01.fits
        if not RampList:
            logging.info('No images to process in {0}'.format(InputDir))
            continue # skip to next directory
        noNDR = None

        ExtraHeaderCalculator = READOUT_SOFTWARE[Config['ReadoutSoftware']]['ExtraHeaderCalculations_func']

        if args.NoNDR_Drop_G is None:
            estimate_NoNDR_Drop_G_func = READOUT_SOFTWARE[Config['ReadoutSoftware']]['estimate_NoNDR_Drop_G_func']
            if estimate_NoNDR_Drop_G_func is not None:
                noNDR,noDrop,noG = estimate_NoNDR_Drop_G_func(imagelist)
        else:
            noNDR,noDrop,noG = tuple([int(i) for i in args.NoNDR_Drop_G.split(':')])

        # Do sanity check that all the expected NDRS are available
        if noNDR is not None:
            ExpectedFramesPerRamp = noNDR*noG
            if Config['ReadoutSoftware'] == 'TeledyneWindows':
                RampTime = fits.getval(imagelist[0],'FRMTIME')*(noNDR+noDrop)*noG
                ExtraHeaderCalculator = partial(ExtraHeaderCalculator,Ramptime=RampTime)
        else:
            ExpectedFramesPerRamp = None

        RampFilenameString = READOUT_SOFTWARE[Config['ReadoutSoftware']]['RampFilenameString']
        FileNameSortKeyFunc = READOUT_SOFTWARE[Config['ReadoutSoftware']]['filename_sort_func']
        SlopeimageGenerator = partial(generate_slope_image,
                                      InputDir=InputDir,OutputDir=OutputDir,
                                      Config = Config,
                                      OutputFileFormat=Config['OutputFileFormat'],
                                      FirstNDR = args.FirstNDR, LastNDR = args.LastNDR,
                                      RampFilenameString = RampFilenameString,
                                      FilenameSortKeyFunc = FileNameSortKeyFunc,
                                      ExtraHeaderDictFunc= ExtraHeaderCalculator)

        logging.info('Output slope images will be written to {0}'.format(OutputDir))
        if ExpectedFramesPerRamp is not None:
            SelectedRampList = []
            for Ramp in RampList:
                NoofImages = len([f for f in  imagelist  if RampFilenameString.format(Ramp) in os.path.basename(f)])
                if  NoofImages == ExpectedFramesPerRamp:
                    SelectedRampList.append(Ramp)
                else:
                    logging.warning('Skipping Incomplete data for Ramp {0}'.format(Ramp))
                    logging.warning('Expected : {0} frames; Found {1} frames'.format(ExpectedFramesPerRamp,NoofImages))
        else:
            SelectedRampList = RampList

        logging.info('Calculating slope for {0} Ramps in {1}'.format(len(SelectedRampList),InputDir))

        # To Run all in a single process serially. Very useful for debugging
        if args.noCPUs == 1:
            for Ramp in SelectedRampList:
                SlopeimageGenerator(Ramp)
            logging.info('Finished {0}'.format(InputDir))
            continue  # Continue with next InputDir

        # Make all the subprocesses inside the pool to ignore SIGINT
        original_sigint_handler = signal.signal(signal.SIGINT,signal.SIG_IGN)
        pool = Pool(processes=args.noCPUs)
        # Make the parent process catch SIGINT by restoring
        signal.signal(signal.SIGINT,original_sigint_handler)
        try:
            outputfiles = pool.map_async(SlopeimageGenerator,SelectedRampList)
            MaximumRunTime = 2*24*60*60 # Two days
            outputfiles.get(MaximumRunTime) # Wait till everything is over
        except KeyboardInterrupt:
            logging.critical('SIGINT Keyboard Interrupt Recevied... Shutting down the script..')
            pool.terminate()
        except TimeoutError:
            logging.critical('TIMEOUT Error. Shutting down the script. (Timeout ={0}s)'.format(MaximumRunTime))
            pool.terminate()
        else:
            pool.close()
            logging.info('Finished {0}'.format(InputDir))



if __name__ == "__main__":
    main()
    
