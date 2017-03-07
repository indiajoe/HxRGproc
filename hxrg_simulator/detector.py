""" This file contains the detector object which simulates HxRG """
from __future__ import division

import detector_params as DP
import nghxrg as ng
from astropy.io import fits
import numpy as np
from scipy import ndimage


class detector(object):
    """ This object is the HxRG detector.
    methods:
        take_dark: returns a dark cube readout
        take_exposure: returns an exposure readout of incidence light
    """
    def __init__(self,**inp_kargs):
        """ Initialises the Noise model object of detector 
        See all allowed input parameters of ng.HXRGNoise for available options.
        By default, uses the configuration in detector_params.py config file
        """
        # update the default config variables with input arguments
        self.DP = DP
        self.update_HXRGNoise(**inp_kargs)

    def update_HXRGNoise(self,**inp_kargs):
        """ Updates the HXRG Noise parameters and the Noise model object """
        self.DP.HXRGNoise_kargs.update(inp_kargs)
        # Recreate the Noise generator object with new parameters
        self.ng_hxrg_cube = ng.HXRGNoise(**self.DP.HXRGNoise_kargs)

    @property
    def t_framereadout(self):
        """ Frame read out speed : Default # 10.7368  s/frame"""
        P = self.DP.HXRGNoise_kargs # parameter dictionary
        return P['dt']* (P['naxis1']+P['nfoh'])* (P['naxis2']//P['n_out'] +P['nroh']) 

    def t_pixelwise_framereadout(self):
        """ Returns a 2D array of actual integration time in each pixel duuring a readout """
        No_of_strips = self.DP.HXRGNoise_kargs['n_out'] # No of detector readout channels
        naxis1 = self.DP.HXRGNoise_kargs['naxis1']
        naxis2 = self.DP.HXRGNoise_kargs['naxis2'] // No_of_strips
        dt = self.DP.HXRGNoise_kargs['dt'] # pixel readout time
        row_overhead = self.DP.HXRGNoise_kargs['nroh'] # number of row overhead gap in pixel units
        Strip_itime_full = np.reshape(np.arange( naxis1 * (naxis2+row_overhead)) * dt,
                                      (naxis1, (naxis2+row_overhead)))

        return np.hstack( [Strip_itime_full[:,:-row_overhead]] *No_of_strips )


    def apply_non_linearity(self,datacube):
        """ Applies non linearity to the data cube """
        # To be implemented later
        print('Non linearity not applied..')
        return datacube
        
    def _calculate_datacube(self,flux_rate):
        """ Returns the readout datacube for input flux_rate on array. 
        Input: 
             flux_rate: 3d effective flux rate array cube, binned in X,Y pixels and time.
        """
        Noise_cube = self.ng_hxrg_cube.mknoise(None,**self.DP.mknoise_kargs).data
        # Get First frame readout's non uniform exposure time
        First_t_frame = self.t_pixelwise_framereadout()
        First_readout = np.random.poisson(flux_rate[0,:,:] * First_t_frame)
        Flux_cube = np.zeros(flux_rate.shape)
        Flux_cube[0,:,:] = First_readout
        
        # Now calculate flux readout for remaining frames
        # First fill the array with delat flux values
        Flux_cube[1:,:,:] = np.random.poisson(flux_rate[1:,:,:] * self.t_framereadout)
        # Now do a cumilative sum to get actual readout
        Flux_cube = np.cumsum(Flux_cube,axis=0)

        # Apply Inter-pixel capacitance convolution
        if isinstance(self.DP.ipc_matrix,np.ndarray):
            for i in range(Flux_cube.shape[0]):
                Flux_cube[i,:,:] = ndimage.convolve(Flux_cube[i,:,:],
                                                    self.DP.ipc_matrix, 
                                                    mode='constant', cval=0.0)
        else:
            print('Ignoring Inter-pixel capacitance..')

        # Apply Non linearity
        FinalDatCube = self.apply_non_linearity(Noise_cube + Flux_cube )
    
        return FinalDatCube

    def set_reference_pixels_zero(self,fluxcube):
        """ Sets the flux in Reference pixel on the edges to Zero for full HxRG 3d readout """
        P = self.DP.HXRGNoise_kargs # parameter dictionary
        # Currently implement only for H2RG full window readout of 2k
        if (P['naxis1'] == 2048):
            fluxcube[:,:,:4] = 0
            fluxcube[:,:,-4:] = 0
        if (P['naxis2'] == 2048):
            fluxcube[:,:4,:] = 0
            fluxcube[:,-4:,:] = 0

        return fluxcube
            

    def take_dark(self,itime=0,outputfile=None):
        """ Returns dark exposure data cube at self.DP.dark_current rate.
        Input:
             itime: total integration time in seconds.
                    Actual exposure time will be rounded up to integer readout time
             outputfile :  optional file name to write a copy of output as fits image
        Output:
             Dark_data_cube: ndarray of dark readout cube 
        """
        # Estimate number of Non distructive readout (NDRs)
        No_of_NDRs = int(itime/self.t_framereadout) + 1
        self.update_HXRGNoise(naxis3 = No_of_NDRs)

        P = self.DP.HXRGNoise_kargs # parameter dictionary
        dark_rate_cube = self.DP.dark_current * np.ones((P['naxis3'],P['naxis2'],P['naxis1']))
        
        # Calculate darkcube
        Dark_data_cube = self._calculate_datacube(self.set_reference_pixels_zero(dark_rate_cube))
        
        if outputfile is not None:
            hduout = fits.PrimaryHDU(Dark_data_cube)
            hduout.writeto(outputfile, clobber=True)
            
        return Dark_data_cube

    def take_exposure(self,incident_flux_const,incident_flux_var=0,flux_scale= None,itime=0,outputfile=None):
        """ Returns exposure data cube when photons hit at incident_flux * flux_scale(time)
        Input:
             incident_flux_const: The constant flux [unit: rate at which electrons are getting created in each pixel (e-/s/pix).]
                          User should multiply with gain and QE for getting this input from photon rate
             incident_flux_var: The time variable flux [unit: rate at which electrons are getting created in each pixel (e-/s/pix).]
                          User should multiply with gain and QE for getting this input from photon rate

             flux_scale(time): function which returns the scale factor for the flux per pixel 
                               as a function of time. This will be multiplied to incident_flux_var
                         default is a constant aperture scaling of 1 for the flux rate
                         ie. flux_scale = lambda x:np.ones(x.shape)

             itime: total integration time in seconds.
                    Actual exposure time will be rounded up to integer readout time
             outputfile :  optional file name to write a copy of output as fits image.
        Output:
             Data_cube: ndarray of exposure readout cube 
        """
        if flux_scale is None: # Use a constant aperture scaling of 1 for the flux rate
            flux_scale = lambda x : np.ones(x.shape)

        # Estimate number of Non distructive readout (NDRs)
        No_of_NDRs = int(itime/self.t_framereadout) + 1
        self.update_HXRGNoise(naxis3 = No_of_NDRs)

        P = self.DP.HXRGNoise_kargs # parameter dictionary
        # First fill the itime array with delta time
        itime_perpixel = np.ones((P['naxis3'],P['naxis2'],P['naxis1'])) * self.t_framereadout
        # First readout is non uniform itime
        itime_perpixel[0,:,:] = self.t_pixelwise_framereadout()
        # cumultatively add the effective itime for each pixel along time axis
        itime_perpixel = np.cumsum(itime_perpixel,axis=0)

        flux_rate_cube = incident_flux_const + incident_flux_var * flux_scale(itime_perpixel)
        
        Effective_eflux = self.set_reference_pixels_zero(flux_rate_cube + self.DP.dark_current)
        # Calculate output data cube
        Data_cube = self._calculate_datacube(Effective_eflux)
        
        if outputfile is not None:
            hduout = fits.PrimaryHDU(Data_cube)
            hduout.writeto(outputfile, clobber=True)
            
        return Data_cube
            
        
