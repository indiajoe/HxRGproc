#!/usr/bin/env python
""" This script is to create slope images from the Up the ramp images"""
import sys
import argparse
from astropy.io import fits
import numpy as np
import numpy.ma
import os
import re
from datetime import datetime
from astropy.time import Time, TimezoneInfo
import astropy.units as u
from multiprocessing import Pool
from functools import wraps, partial
import logging
from . import reduction 

SaturationCount = 50000  # TODO: Move these to config files later

def LogMemoryErrors(func):
    """ This is a decorator to log Memory errors """
    @wraps(func)
    def wrappedFunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError as e:
            logging.critical('Memory Error: Please free up Memory on this computer to continue..')
            logging.error(e)
            logging.info('Hint: Reduce number of processes running in parallel on this computer')
            raise
    return wrappedFunc

def LoadDataCube(filelist):
    """ Loads the input file list in to a data cube, and return both datacube and fits header seperately.
    Input: 
         filelist : List of sorted filenames of the up-the-ramp data
    Returns:
         DataCube, fitsheader of first img
    """
    logging.info('Loading UTR: {0}'.format(sorted(filelist)))
    # datalist = [subtractReferencePixels(fits.getdata(f)) for f in sorted(filelist)]
    datalist = [fits.getdata(f) for f in sorted(filelist)]
    return np.array(datalist), fits.getheader(filelist[0])


def calculate_slope_image(UTRlist):
    """ Returns slope image hdulist object, and header from the input up-the-ramp fits file list 
    Input:
          UTRlist: List of up-the-ramp NDR files
    Returns:
         Slopeimage hdulist object, header
"""
    time = np.array([fits.getval(f,'INTTIME') for f in UTRlist])
    DataCube, header = LoadDataCube(UTRlist) 
    NoNDR = DataCube.shape[0]
    PedestalSubSaturation = SaturationCount - np.median(DataCube[0,:,:])
    DataCube = reduction.remove_biases_in_cube(DataCube,time=time,no_channels=header['NOUTPUTS'],do_LSQmedian_correction=True)
    logging.info('Number of NDRs = {0}'.format(NoNDR))
    logging.info('Fitting slope..')
    slopeimg,alpha = reduction.slope_img_from_cube(np.ma.masked_greater(DataCube,
                                                                        PedestalSubSaturation), 
                                                   time)
    # convert all masked values to nan
    slopeimg = np.ma.filled(slopeimg,fill_value=np.nan)
    header['NoNDR'] = (NoNDR, 'No of NDRs used in slope')
    header['history'] = 'Slope image generated'
    hdu = fits.PrimaryHDU(slopeimg,header=header)
    hdulist = fits.HDUList([hdu])
    return hdulist,header

@LogMemoryErrors
def generate_slope_image(RampNo,InputDir,OutputDir,OutputFilePrefix='Slope-R',
                         FirstNDR = 0, LastNDR = None,
                         RampFilenamePrefix='H2RG_R{0:02}_M',
                         FilenameSortKeyFunc = None,
                         ExtraHeaderDictFunc= None):
    """ 
    Generates the Slope image and writes the output slope image for the data files.
    Input:
        RampNo: (int) Ramp number (Ex: 1)
        InputDir: (str) Input Directory containing the UTR files
        OutputDir: (str) Output Directory to write output slope images into
        OutputFilePrefix: (str,optional) Output file prefix (default: 'Slope-R')
        FirstNDR = (int,optional) Number of initial NDRs to skip for slope fitting (default: 0) 
        LastNDR = (int/None,optional) Number of the maximum NDR to use in slope fitting (default: None means all)
        RampFilenamePrefix= (str,optional) Filenmae format which uniquely identifies a particular Ramp file (Default: 'H2RG_R{0:02}_M')
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

    OutputFileName = os.path.join(OutputDir,OutputFilePrefix+'{0}.fits'.format(RampNo))
    try:
        os.makedirs(OutputDir)
    except OSError as e:
        logging.info(e)
        logging.info('Ignore WARNING : Output dir exists. ')

    if os.path.isfile(OutputFileName):
        logging.warning('WARNING: Output file {0} already exist...'.format(OutputFileName))
        logging.warning('Skipping regeneration of the slope image')
        return None
        
    # First search for all raw files
    UTRlistT = sorted((os.path.join(InputDir,f) for f in os.listdir(InputDir) if (os.path.splitext(f)[-1] == '.fits') and (RampFilenamePrefix.format(RampNo) in os.path.basename(f))))
    UTRlist = sorted(UTRlistT,key=FilenameSortKeyFunc)

    if LastNDR is None:
        LastNDR = len(UTRlist)
    logging.info('Loading Ramp data of {0}'.format(RampNo))
    Slopehdulist,header = calculate_slope_image(UTRlist[FirstNDR:LastNDR])
    if ExtraHeaderDictFunc is not None:
        for key,value in ExtraHeaderDictFunc(header).items():
            Slopehdulist[0].header[key] = value

    Slopehdulist.writeto(OutputFileName)
    return OutputFileName


#### Functions specific to help reduce Windows Teledyne software data
def TeledyneFileNameSortKeyFunc(fname):
    """ Function which returns the key to sort Teledyne filename """
    return tuple(map(int,re.search('H2RG_R(.+?)_M(.+?)_N(.+?).fits',os.path.basename(fname)).group(1,2,3)))

def ExtraHeaderCalculations4Windows(header,Ramptime):
    """ Returns a dictionary of extra entires for slope header """
    utc_minus_four_hour = TimezoneInfo(utc_offset=-4*u.hour)
    month2nub ={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    ExtraHeader = {}

    ExtraHeader['OBSTIME'] = (header['ACQTIME'] + ((header['SEQNUM_R']*header['NRESETS']*header['FRMTIME']) + ((header['SEQNUM_R']-1)*Ramptime))/(60*60*24.0), 'Estimated Observation Time')
    # Because of this silly SIMPLE header (we need for calculating file write time), we need raw fits file header. Don't use hdulist's header.
    t = Time(datetime(*tuple([int(header.comments['SIMPLE'].split()[-1]),month2nub[header.comments['SIMPLE'].split()[-4]],
                              int(header.comments['SIMPLE'].split()[-3])]+map(int,header.comments['SIMPLE'].split()[-2].split(':'))),tzinfo=utc_minus_four_hour))
    ExtraHeader['FWTIME'] = (t.jd,'Time raw fits image was written')

    return ExtraHeader


def parse_args_Teledyne():
    """ Parses the command line input arguments for Teledyne Software data reduction"""
    parser = argparse.ArgumentParser(description="Script to Generate Slope/Flux images from Up-the-Ramp data taken using Teledyne's Windows software")
    parser.add_argument('InputDir', type=str,
                        help="Input Directory contiaining the Up-the-Ramp Raw data files")
    parser.add_argument('OutputMasterDir', type=str,
                        help="Output Master Directory to which output images to be written")
    parser.add_argument('NoNDR_Drop_G', type=str, 
                        help="No of NDRS per Group:No of Drops:No of Groups (Example  40:60:5)")
    parser.add_argument('--FirstNDR', type=int, default=0,
                        help="Number of First NDRs to be skipped")
    parser.add_argument('--LastNDR', type=int, default=None,
                        help="Maximum NDRs to be used. (Default: all)")
    parser.add_argument('--noCPUs', type=int, default=1,
                        help="Number of parallel CPUs to be used to process independent Ramps in parallel")
    parser.add_argument('--logfile', type=str, default=None,
                        help="Log Filename to write logs during the run")

    args = parser.parse_args()
    return args
    
def main_Teledyne():
    """ Standalone Script to generate Slope images from Up the Ramp data taken using Teledyne's Windows software"""
    args = parse_args_Teledyne()    

    if args.logfile is None:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO, 
                            filename=args.logfile, filemode='a')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # Sent info to the stdout as well


    OutputDir = os.path.join(args.OutputMasterDir,os.path.basename(args.InputDir.rstrip('/')))
    RampFilenamePrefix='H2RG_R{0:02}_M'

    # Find the number of Ramps in the input Directory
    imagelist = sorted((os.path.join(args.InputDir,f) for f in os.listdir(args.InputDir) if (os.path.splitext(f)[-1] == '.fits')))
    RampList = sorted(set((int(re.search('H2RG_R(.+?)_M',os.path.basename(f)).group(1)) for f in imagelist))) # 45 in H2RG_R45_M01_N01.fits

    # Do sanity check that all the expected NDRS are avialable
    noNDR,noDrop,noG = tuple([int(i) for i in args.NoNDR_Drop_G.split(':')])
    ExpectedFramesPerRamp = noNDR*noG
    RampTime = fits.getval(imagelist[0],'FRMTIME')*(noNDR+noDrop)*noG
    TeledyneExtraHeaderCalculator = partial(ExtraHeaderCalculations4Windows,Ramptime=RampTime)

    TeledyneWindowsSlopeimageGenerator = partial(generate_slope_image,
                                                 InputDir=args.InputDir,OutputDir=OutputDir,OutputFilePrefix='Slope-R',
                                                 FirstNDR = args.FirstNDR, LastNDR = args.LastNDR,
                                                 RampFilenamePrefix=RampFilenamePrefix,
                                                 FilenameSortKeyFunc = TeledyneFileNameSortKeyFunc,
                                                 ExtraHeaderDictFunc= TeledyneExtraHeaderCalculator)

    logging.info('Processing data in {0}'.format(args.InputDir))
    logging.info('Output slope images will be written to {0}'.format(OutputDir))
    SelectedRampList = []
    for Ramp in RampList:
        NoofImages = len([f for f in  imagelist  if RampFilenamePrefix.format(Ramp) in os.path.basename(f)])
        if  NoofImages == ExpectedFramesPerRamp:
            SelectedRampList.append(Ramp)
        else:
            logging.warning('Skipping Incomplete data for Ramp {0}'.format(Ramp))
            logging.warning('Expected : {0} frames; Found {1} frames'.format(ExpectedFramesPerRamp,NoofImages))
            
    pool = Pool(processes=args.noCPUs)
    pool.map(TeledyneWindowsSlopeimageGenerator,SelectedRampList)
    # for Ramp in SelectedRampList:
    #     TeledyneWindowsSlopeimageGenerator(Ramp)

###############################################

if __name__ == "__main__":
    main_Teledyne()
    
