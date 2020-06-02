#!/usr/bin/env python
""" This script is to create naive cds images from the Up the ramp images"""
from __future__ import division
import sys
import argparse
from astropy.io import fits
import numpy as np
import numpy.ma
import os
import re
import errno
from multiprocessing import TimeoutError
from multiprocessing.pool import Pool
import logging
import signal
import traceback
from . import reduction 
from .generate_slope_images import estimate_NoNDR_Drop_G_TeledyneData, FileNameSortKeyFunc_Teledyne, FileNameSortKeyFunc_HPFLinux, pack_traceback_to_errormsg, log_all_uncaughtexceptions_handler, LogMemoryErrors
try:
    import ConfigParser
    from functools32 import wraps, partial
except ModuleNotFoundError:  # Python 3 environment
    import configparser as ConfigParser
    from functools import wraps, partial


def calculate_cds_image(FirstImage,LastImage):
    """ Returns the LastImage-FirstImage data hdulist and header """
    FirstImageArray = fits.getdata(FirstImage).astype(np.float)  
    LastImageArray = fits.getdata(LastImage).astype(np.float)  
    CDSimage = LastImageArray - FirstImageArray
    header = fits.getheader(FirstImage)
    # Do reference pixel subtraction
    CDSimage = reduction.subtract_reference_pixels(CDSimage,no_channels=header['CHANNELS'])
    header['history'] = 'CDS image generated from {1}-{0}'.format(FirstImage,LastImage)
    hdu = fits.PrimaryHDU(CDSimage,header=header)
    hdulist = fits.HDUList([hdu])
    return hdulist, header


# @pack_traceback_to_errormsg can be commented out when function not used in multiprocess
@pack_traceback_to_errormsg
@LogMemoryErrors
def generate_cds_image(RampNo,InputDir,OutputDir,OutputFilePrefix='CDS-R',
                       FirstNDR=0,LastNDR=None,
                       RampFilenamePrefix='H2RG_R{0:02}_M',
                       FilenameSortKeyFunc = None):
    """ 
    Generates the CDS images and writes the output CDS image for the data files.
    """
    if FilenameSortKeyFunc is None:
        # Use the file name string itself to sort!
        logging.warning('No function to sort filename provided, using default filename sorting..')
        FilenameSortKeyFunc = lambda f: f

    OutputFileName = os.path.join(OutputDir,OutputFilePrefix+'{0}.fits'.format(RampNo))
    try:
        os.makedirs(OutputDir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            logging.info(e)
            raise
        pass # Ignore the error that Output dir already exist.

    if os.path.isfile(OutputFileName):
        logging.warning('WARNING: Output file {0} already exist...'.format(OutputFileName))
        logging.warning('Skipping regeneration of the CDS image')
        return None
        
    # First search for all raw files
    UTRlistT = sorted((os.path.join(InputDir,f) for f in os.listdir(InputDir) if (os.path.splitext(f)[-1] == '.fits') and (RampFilenamePrefix.format(RampNo) in os.path.basename(f))))
    UTRlist = sorted(UTRlistT,key=FilenameSortKeyFunc)

    if LastNDR is None:
        LastNDR = len(UTRlist)
    logging.info('Creating CDS of {0}'.format(RampNo))
    CDShdulist,header = calculate_cds_image(UTRlist[FirstNDR],UTRlist[LastNDR-1])
    CDShdulist.writeto(OutputFileName)
    return OutputFileName


def parse_args():
    """ Parses the command line input arguments for CDS data reduction of up-the-ramp"""
    parser = argparse.ArgumentParser(description="Script to Generate CDS images from Up-the-Ramp")
    parser.add_argument('InputDir', type=str,
                        help="Input Directory contiaining the Up-the-Ramp Raw data files. Multiple directories can be provided comma seperated.")
    parser.add_argument('OutputMasterDir', type=str,
                        help="Output Master Directory to which output CDS images to be written")
    parser.add_argument('Instrument', choices=ReadOutSoftware.keys(),
                        help="Name of the supported instrument data")
    parser.add_argument('--NoNDR_Drop_G', type=str, default=None,
                        help="No of NDRS per Group:No of Drops:No of Groups (Example  40:60:5)")
    parser.add_argument('--FirstNDR', type=int, default=0,
                        help="Number of First NDRs to be skipped")
    parser.add_argument('--LastNDR', type=int, default=None,
                        help="Maximum NDRs to be used. (Default: last)")
    parser.add_argument('--noCPUs', type=int, default=1,
                        help="Number of parallel CPUs to be used to process independent Ramps in parallel")
    parser.add_argument('--logfile', type=str, default=None,
                        help="Log Filename to write logs during the run")
    parser.add_argument("--loglevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help="Set the logging level")

    args = parser.parse_args()
    return args


def main():
    """ Standalone Script to generate CDS images from Up the Ramp data"""
    # Override the default exception hook with our custom handler
    sys.excepthook = log_all_uncaughtexceptions_handler

    args = parse_args()    

    INST = args.Instrument
    if INST not in ReadOutSoftware:
        logging.error('Instrument {0} not supported'.format(INST))
        sys.exit(1)

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

        OutputDir = os.path.join(args.OutputMasterDir,os.path.basename(InputDir.rstrip('/')))
        RampFilenamePrefix = ReadOutSoftware[INST]['RampFilenameString']

        InputDir = os.path.join(InputDir,ReadOutSoftware[INST]['InputSubDir']) # Append any redundant input subdirectory to be added

        # Find the number of Ramps in the input Directory
        imagelist = sorted((os.path.join(InputDir,f) for f in os.listdir(InputDir) if (os.path.splitext(f)[-1] == '.fits')))
        RampList = sorted(set((re.search(ReadOutSoftware[INST]['RampidRegexp'],os.path.basename(f)).group(1) for f in imagelist))) # 45 in H2RG_R45_M01_N01.fits
        if not RampList:
            logging.info('No images to process in {0}'.format(InputDir))
            continue # skip to next directory

        noNDR = None
        if args.NoNDR_Drop_G is None:
            if ReadOutSoftware[INST]['estimate_NoNDR_Drop_G_func'] is not None:
                noNDR,noDrop,noG = ReadOutSoftware[INST]['estimate_NoNDR_Drop_G_func'](imagelist)
        else:
            noNDR,noDrop,noG = tuple([int(i) for i in args.NoNDR_Drop_G.split(':')])

        # Do sanity check that all the expected NDRS are available
        ExpectedFramesPerRamp = noNDR*noG if (noNDR is not None) else None


        CDSImageGenerator = partial(generate_cds_image,
                                    InputDir = InputDir,
                                    OutputDir = OutputDir,
                                    OutputFilePrefix = 'CDS-R',
                                    FirstNDR = args.FirstNDR, LastNDR = args.LastNDR,
                                    RampFilenamePrefix = RampFilenamePrefix,
                                    FilenameSortKeyFunc = ReadOutSoftware[INST]['filename_sort_func'])


        logging.info('Output CDS images will be written to {0}'.format(OutputDir))
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

        logging.info('Calculating CDS for {0} Ramps in {1}'.format(len(SelectedRampList),InputDir))

        # # To Run all in a single process serially. Very useful for debugging
        # for Ramp in SelectedRampList:
        #     CDSImageGenerator(Ramp)
        # Make all the subprocesses inside the pool to ignore SIGINT
        original_sigint_handler = signal.signal(signal.SIGINT,signal.SIG_IGN)
        pool = Pool(processes=args.noCPUs)
        # Make the parent process catch SIGINT by restoring
        signal.signal(signal.SIGINT,original_sigint_handler)
        try:
            outputfiles = pool.map_async(CDSImageGenerator,SelectedRampList)
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

###############################################

# Instrument specific configuration
# Register functions which are specific to each readout software output in dictionary below
ReadOutSoftware = {
    'TeledyneWindows':{'RampFilenameString' : 'H2RG_R{0}_M', #Input filename structure with Ramp id substitution
                       'RampidRegexp' : 'H2RG_R(.+?)_M', # Regexp to extract unique Ramp id from filename
                       'InputSubDir' : '', # Append any redundant input subdirectory to be added

                       'filename_sort_func' : FileNameSortKeyFunc_Teledyne,
                       'estimate_NoNDR_Drop_G_func' : estimate_NoNDR_Drop_G_TeledyneData},

    'HPFLinux':{'RampFilenameString' : 'hpf_{0}_F', #Input filename structure with Ramp id substitution
                'RampidRegexp' : 'hpf_(.*_R\d*?)_F.*fits', # Regexp to extract unique Ramp id from filename
                'InputSubDir' : 'fits', # Append any redundant input subdirectory to be added
                'filename_sort_func': FileNameSortKeyFunc_HPFLinux,
                'estimate_NoNDR_Drop_G_func':None},
    }


if __name__ == "__main__":
    main()
