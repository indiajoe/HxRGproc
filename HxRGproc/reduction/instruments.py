#!/usr/bin/env python
""" This module contains the instrument specific functions """
import os
import re
import logging
import numpy as np
from astropy.time import Time, TimezoneInfo
from astropy.io import fits
from scipy.interpolate import interp1d

#####################################################################
#### Functions specific to help reduce Windows Teledyne software data
#####################################################################
def sort_filename_key_function_Teledyne(fname):
    """ Function which returns the key to sort Teledyne filename """
    return tuple(map(int,re.search('H2RG_R(.+?)_M(.+?)_N(.+?).fits',os.path.basename(fname)).group(1,2,3)))

def extra_header_calculations_Teledyne(header,Ramptime):
    """ Returns a dictionary of extra entires for slope header """
    utc_minus_four_hour = TimezoneInfo(utc_offset=-4*u.hour)
    month2nub ={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    ExtraHeader = {}

    ExtraHeader['OBSTIME'] = (header['ACQTIME'] + ((header['SEQNUM_R']*header['NRESETS']*header['FRMTIME']) + ((header['SEQNUM_R']-1)*Ramptime))/(60*60*24.0), 'Estimated Observation Time')
    # Because of this silly SIMPLE header (we need for calculating file write time), we need raw fits file header. Don't use hdulist's header.
    t = Time(datetime(*tuple([int(header.comments['SIMPLE'].split()[-1]),month2nub[header.comments['SIMPLE'].split()[-4]],
                              int(header.comments['SIMPLE'].split()[-3])]+list(map(int,header.comments['SIMPLE'].split()[-2].split(':')))),tzinfo=utc_minus_four_hour))
    ExtraHeader['FWTIME'] = (t.jd,'Time raw fits image was written')

    return ExtraHeader

def estimate_NoNDR_Drops_G_Teledyne(imagelist):
    """ Returns (No of NDRs in Group, Drops, number of Groups) based on the imagename list """
    RampList = sorted(set((int(re.search('H2RG_R(.+?)_M',os.path.basename(f)).group(1)) for f in imagelist))) # 45 in H2RG_R45_M01_N01.fits
    GroupList = sorted(set((int(re.search('H2RG_R.+?_M(.+?)_',os.path.basename(f)).group(1)) for f in imagelist))) # 5 in H2RG_R01_M05_N01.fits
    noNDRList = sorted(set((int(re.search('H2RG_R.+?_M.+?_N(.+?).fits',os.path.basename(f)).group(1)) for f in imagelist))) # 1 in H2RG_R45_M05_N01.fits
    noG = max(GroupList)
    noR = max(RampList)
    noNDR = max(noNDRList)
    if noG > 1:
        # We need to estimate the Drops between frames, which is not recorded anywhere in headers by Teledyne softwatre
        ImageDir = os.path.dirname(imagelist[0])
        frametime = fits.getval(imagelist[0],'FRMTIME')
        itimeLastNDRinG01 = fits.getval(os.path.join(ImageDir,'H2RG_R{0:02}_M01_N{1:02}.fits'.format(min(RampList),noNDR)),'INTTIME')
        itimeFirstNDRinG02 = fits.getval(os.path.join(ImageDir,'H2RG_R{0:02}_M02_N01.fits'.format(min(RampList))),'INTTIME')
        NoOfDrops = int(round((itimeFirstNDRinG02 - itimeLastNDRinG01)/frametime))-1
    else:
        NoOfDrops = 0 # is irrelevant

    logging.info('Estimated Number of (NDRs in Group: Drops: Groups) = {0}:{1}:{2}'.format(noNDR,NoOfDrops,noG))
    return noNDR,NoOfDrops,noG

#####################################################################
#### Functions specific to reduce HPFLinux software data
#####################################################################
def sort_filename_key_function_HPFLinux(fname):
    """ Function which returns the key to sort HPFLinux filename """
    return tuple(map(int,re.search('hpf_(\d+?)T(\d+?)_R(\d+?)_F(\d+?).fits',os.path.basename(fname)).group(1,2,3,4)))

def fix_header_function_HPFLinux(header,fname=None):
    """ Funtion to fix any missing headers needed in header """
    if 'ITIME' not in header:
        try:
            FrameTime = header['FRMTIME']
        except KeyError:
            if header['CHANNELS'] == 4:
                FrameTime = 10.65 # 10.48576 
            elif header['CHANNELS'] == 32:
                FrameTime = 10.65/8. 
            else:
                raise
        
        header['ITIME'] = header['FRAMENUM']*FrameTime
    return header

def fix_datacube_function_HPFLinux(DataCube):
    """Fixes the zero readout rows in datacube """
    if np.any(DataCube[:,:,-1] == 0) : # Fix last blank column
        DataCube[:,:,-1] = DataCube[:,:,-2]

    ZeroMask = DataCube == 0
    IJZeroMask = np.any(ZeroMask,axis=0)
    t = np.arange(DataCube.shape[0])
    for i,j in zip(*np.where(IJZeroMask)):
        try:
            f = interp1d(t[~ZeroMask[:,i,j]],DataCube[:,i,j][~ZeroMask[:,i,j]],kind='linear',fill_value='extrapolate')
        except ValueError as e:
            logging.error(e)
            logging.error('Unable to fix the Zero in {0},{1} pix due to lack of enough non-zero data {2}'.format(i,j,len(t[~ZeroMask[:,i,j]])))
        else:
            DataCube[:,i,j][ZeroMask[:,i,j]] = f(t[ZeroMask[:,i,j]])

    return DataCube
#####################################################################
#####################################################################
#### Functions specific to reduce SpecTANSPEC software data
#####################################################################

def sort_filename_key_function_SpecTANSPEC(fname):
    """ Function which returns the key to sort SpecTANSPEC filename """
    return tuple(map(int,re.search('.*-(\d+?)\.Z\.(\d+?)\.fits',os.path.basename(fname)).group(1,2)))


def fix_header_function_SpecTANSPEC(header,fname=None):
   """Function to fix any missing headers needed in header"""
   if 'CHANNELS' not in header:
           header['CHANNELS'] = 4
   if ('NDRITIME' not in header) or (header['NDRITIME'] == 0):
       FrameTime = 5.253  #Time taken for each readout.
       Frame_Number = re.search('.*\.Z\.(\d+?)\.fits',os.path.basename(fname)).group(1)
       time = int(Frame_Number) * FrameTime
       header['NDRITIME'] = time
   return header

def fix_datacube_function_SpecTANSPEC(DataCube):
    """Fixes the zero readout rows in datacube"""
    DataCube = 65536-DataCube.astype(np.float32)
    return DataCube

#####################################################################
#####################################################################
#### Functions specific to reduce TIRSPEC software data
#####################################################################

def sort_filename_key_function_TIRSPEC(fname):
    """ Function which returns the key to sort TIRSPEC filename """
    return tuple(map(int,re.search('.*-(\d+?)-debug-(\d+?)\.fits',os.path.basename(fname)).group(1,2)))


def fix_header_function_TIRSPEC(header,fname=None):
   """Function to fix any missing headers needed in header"""
   if 'CHANNELS' not in header:
           header['CHANNELS'] = 16
   if ('NDRITIME' not in header) or (header['NDRITIME'] == 0):
       FrameTime = header['NDRRDTM']  #Time taken for each readout.
       Frame_Number = re.search('.*debug-(\d+?)\.fits',os.path.basename(fname)).group(1)
       time = int(Frame_Number) * FrameTime
       header['NDRITIME'] = time
   return header

def fix_datacube_function_TIRSPEC(DataCube):
    """Fixes the zero readout rows in datacube"""
    DataCube = 65536-DataCube.astype(np.float32)
    return DataCube


####################################################################
# Register functions which are specific to each readout software output in dictionary below
#####################################################################
# For the generate_slope_images.py

SupportedReadOutSoftware_for_slope = {
    'TeledyneWindows':{'RampFilenameString' : 'H2RG_R{0}_M', #Input filename structure with Ramp id substitution
                       'RampidRegexp' : 'H2RG_R(.+?)_M', # Regexp to extract unique Ramp id from filename
                       'HDR_NOUTPUTS' : 'NOUTPUTS', # Fits header for number of output channels
                       'HDR_INTTIME' : 'INTTIME', # Fits header for accumulated exposure time in each NDR
                       'filename_sort_func' : sort_filename_key_function_Teledyne,
                       'FixHeader_func': lambda hdr, fname=None: hdr, # Optional function call to fix input raw header
                       'FixDataCube_func': lambda Dcube: Dcube, # Optional function call to fix input Data Cube
                       'estimate_NoNDR_Drop_G_func' : estimate_NoNDR_Drops_G_Teledyne,
                       'ExtraHeaderCalculations_func' : extra_header_calculations_Teledyne},

    'HPFLinux':{'RampFilenameString' : 'hpf_{0}_F', #Input filename structure with Ramp id substitution
                'RampidRegexp' : 'hpf_(.*_R\d*?)_F.*fits', # Regexp to extract unique Ramp id from filename
                'HDR_NOUTPUTS' : 'CHANNELS', # Fits header for number of output channels
                'HDR_INTTIME' : 'ITIME', # Fits header for accumulated exposure time in each NDR
                'filename_sort_func': sort_filename_key_function_HPFLinux,
                'FixHeader_func': fix_header_function_HPFLinux, # Optional function call to fix input raw header
                'FixDataCube_func': fix_datacube_function_HPFLinux, # Optional function call to fix input Data Cube
                'estimate_NoNDR_Drop_G_func':None,
                'ExtraHeaderCalculations_func':None},
    'HPFMACIE':{'RampFilenameString' : 'hpf_{0}_F', #Input filename structure with Ramp id substitution
                'RampidRegexp' : 'hpf_(.*_R\d*?)_F.*fits', # Regexp to extract unique Ramp id from filename
                'HDR_NOUTPUTS' : 'CHANNELS', # Fits header for number of output channels
                'HDR_INTTIME' : 'ITIME', # Fits header for accumulated exposure time in each NDR
                'filename_sort_func': sort_filename_key_function_HPFLinux,
                'FixHeader_func': lambda hdr, fname=None: hdr, # Optional function call to fix input raw header
                'FixDataCube_func': lambda Dcube: Dcube, # Optional function call to fix input Data Cube
                'estimate_NoNDR_Drop_G_func':None,
                'ExtraHeaderCalculations_func':None},
    'SpecTANSPEC':{'RampFilenameString':'{0}.Z.',#Input filename structure with Ramp id substitution
                   'RampidRegexp':'(.*?-\d*?)\.Z\.\d*\.fits',# Regexp to extract unique Ramp id from filename
                   'HDR_NOUTPUTS' : 'CHANNELS', # Fits header for number of output channels
                   'HDR_INTTIME' : 'NDRITIME', # Fits header for accumulated exposure time in each NDR
                   'filename_sort_func': sort_filename_key_function_SpecTANSPEC,
                   'FixHeader_func': fix_header_function_SpecTANSPEC,
                   'FixDataCube_func': fix_datacube_function_SpecTANSPEC,
                   'estimate_NoNDR_Drop_G_func': None,
                   'ExtraHeaderCalculations_func': None},
    'TIRSPEC':{'RampFilenameString':'{0}-',#Input filename structure with Ramp id substitution
                   'RampidRegexp':'(.*?-\d*?)-debug-\d*\.fits',# Regexp to extract unique Ramp id from filename
                   'HDR_NOUTPUTS' : 'CHANNELS', # Fits header for number of output channels
                   'HDR_INTTIME' : 'NDRITIME', # Fits header for accumulated exposure time in each NDR
                   'filename_sort_func': sort_filename_key_function_TIRSPEC,
                   'FixHeader_func': fix_header_function_TIRSPEC,
                   'FixDataCube_func': fix_datacube_function_TIRSPEC,
                   'estimate_NoNDR_Drop_G_func': None,
                   'ExtraHeaderCalculations_func': None},
    
}

####################################################################
# Register functions which are specific to each readout software output in dictionary below
#####################################################################
# For the generate_cds_images.py

SupportedReadOutSoftware_for_cds = {
    'TeledyneWindows':{'RampFilenameString' : 'H2RG_R{0}_M', #Input filename structure with Ramp id substitution
                       'RampidRegexp' : 'H2RG_R(.+?)_M', # Regexp to extract unique Ramp id from filename
                       'InputSubDir' : '', # Append any redundant input subdirectory to be added
                       'filename_sort_func' : sort_filename_key_function_Teledyne,
                       'estimate_NoNDR_Drop_G_func' : estimate_NoNDR_Drops_G_Teledyne},

    'HPFLinux':{'RampFilenameString' : 'hpf_{0}_F', #Input filename structure with Ramp id substitution
                'RampidRegexp' : 'hpf_(.*_R\d*?)_F.*fits', # Regexp to extract unique Ramp id from filename
                'InputSubDir' : 'fits', # Append any redundant input subdirectory to be added
                'filename_sort_func': sort_filename_key_function_HPFLinux,
                'estimate_NoNDR_Drop_G_func':None},
    'HPFMACIE':{'RampFilenameString' : 'hpf_{0}_F', #Input filename structure with Ramp id substitution
                'RampidRegexp' : 'hpf_(.*_R\d*?)_F.*fits', # Regexp to extract unique Ramp id from filename
                'InputSubDir' : 'fits', # Append any redundant input subdirectory to be added
                'filename_sort_func': sort_filename_key_function_HPFLinux,
                'estimate_NoNDR_Drop_G_func':None},
    'SpecTANSPEC':{'RampFilenameString':'{0}.Z.',#Inpui filename structure with Ramp id substitution
                   'RampidRegexp':'.*-(\d*?)\.Z\.\d*?\.fits',# Regexp to extract unique Ramp id from filename
                   'InputSubDir' : '', # Append any redundant input subdirectory to be added
                   'filename_sort_func':sort_filename_key_function_SpecTANSPEC,
                   'estimate_NoNDR_Drop_G_func':None},
    'TIRSPEC':{'RampFilenameString':'{0}-',#Inpui filename structure with Ramp id substitution
                   'RampidRegexp':'(.*?-\d*?)-debug-\d*?\.fits',# Regexp to extract unique Ramp id from filename
                   'InputSubDir' : '', # Append any redundant input subdirectory to be added
                   'filename_sort_func':sort_filename_key_function_TIRSPEC,
                   'estimate_NoNDR_Drop_G_func':None}
}
