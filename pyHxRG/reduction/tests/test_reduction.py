#!/usr/bin/env python 
""" This is unit tests for functions in reduction/reduction.py """
import unittest
import os
import numpy as np
import cPickle

from pyHxRG.reduction import reduction
from pyHxRG.reduction import generate_slope_images as gsi

TestDataDir = os.path.join(os.path.dirname(__file__), 'testdata')

class BiasCorrectionTests(unittest.TestCase):

    def setUp(self):
        self.RawDataCube, self.RawHeader = self.LoadRawDataForTest()
        self.NoNDR = self.RawDataCube.shape[0]
        self.time = np.arange(self.NoNDR)*10.48576


    def LoadRawDataForTest(self):
        """ Loads the data cube for before beginnign all the tests"""
        InputDir = os.path.join(TestDataDir,'20180420')
        UTRlist = sorted((os.path.join(InputDir,f) for f in os.listdir(InputDir) if (os.path.splitext(f)[-1] == '.fits')))
        print('Loading Test Data set in {0}'.format(InputDir))
        return gsi.LoadDataCube(UTRlist) 

    def test_remove_biases_in_cube(self):
        """ Tests the DataCube level Bias subtraction code """
        print('Testing reduction.remove_biases_in_cube')
        BiasRemovedDataCube = reduction.remove_biases_in_cube(self.RawDataCube,
                                                              time=self.time,
                                                              no_channels=4,
                                                              do_LSQmedian_correction=3000)
        # np.save(os.path.join(TestDataDir,'BiasRemovedDataCubeTestResult.npy'),BiasRemovedDataCube)
        TestResultCube = np.load(os.path.join(TestDataDir,'BiasRemovedDataCubeTestResult.npy'))
        np.testing.assert_array_equal(TestResultCube,BiasRemovedDataCube)

    def test_remove_biases_in_cube_NoLSQM(self):
        """ Tests the DataCube level Bias subtraction code without LSQ median correction"""
        print('Testing reduction.remove_biases_in_cube without LSQ median correction')
        BiasRemovedDataCube = reduction.remove_biases_in_cube(self.RawDataCube,
                                                              time=self.time,
                                                              no_channels=4,
                                                              do_LSQmedian_correction=-9999)
        # np.save(os.path.join(TestDataDir,'BiasRemovedDataCubeTestResult_NoLSQM.npy'),BiasRemovedDataCube)
        TestResultCube = np.load(os.path.join(TestDataDir,'BiasRemovedDataCubeTestResult_NoLSQM.npy'))
        np.testing.assert_array_equal(TestResultCube,BiasRemovedDataCube)

    def test_robust_medianfromPercentiles(self):
        """ Tests the robust median calculation from lower Percentiles """
        print("Testing robust median calculation from lower Percentiles")
        np.testing.assert_almost_equal(reduction.robust_medianfromPercentiles(self.RawDataCube[0,:,:]), 12524.4319439, decimal=7)
        np.testing.assert_almost_equal(reduction.robust_medianfromPercentiles(self.RawDataCube[1,:,:]), 12520.4081537, decimal=7)
        np.testing.assert_almost_equal(reduction.robust_medianfromPercentiles(self.RawDataCube[2,:,:]), 12527.1396087, decimal=7)
        np.testing.assert_almost_equal(reduction.robust_medianfromPercentiles(self.RawDataCube[3,:,:]), 12530.8758138, decimal=7)
        

class SlopeImageGenerationTests(unittest.TestCase):
    def setUp(self):
        self.DataCube = np.load(os.path.join(TestDataDir,'BiasRemovedDataCubeTestResult.npy'))
        self.NoNDR = self.DataCube.shape[0]
        self.time = np.arange(self.NoNDR)*10.48576

    def test_slope_img_from_cube(self):
        """ Test the Slope calculation """
        print('Testing reduction.slope_img_from_cube')
        beta,alpha = reduction.slope_img_from_cube(self.DataCube,self.time)
        # np.save(os.path.join(TestDataDir,'SlopeImageTestBeta.npy'),beta.data)
        # np.save(os.path.join(TestDataDir,'SlopeImageTestAlpha.npy'),alpha.data)
        TestResultAlpha = np.load(os.path.join(TestDataDir,'SlopeImageTestAlpha.npy'))
        TestResultBeta = np.load(os.path.join(TestDataDir,'SlopeImageTestBeta.npy'))
        np.testing.assert_array_equal(TestResultAlpha,alpha.data)
        np.testing.assert_array_equal(TestResultBeta,beta.data)

    def test_varience_of_slope(self):
        """ Test the varience calculation """
        print('Testing reduction.varience_of_slope')
        TestResultSlope = np.load(os.path.join(TestDataDir,'SlopeImageTestBeta.npy'))
        Var = reduction.varience_of_slope(TestResultSlope,self.NoNDR,10.48576,4,2.5)
        # np.save(os.path.join(TestDataDir,'SlopeImageTestVar.npy'),Var)
        TestResultVar = np.load(os.path.join(TestDataDir,'SlopeImageTestVar.npy'))
        np.testing.assert_array_equal(TestResultVar,Var)

    def test_abrupt_change_locations(self):
        """ Test abrupt_change_locations detection function """
        print('Testing reduction.abrupt_change_locations')
        T,I,J = reduction.abrupt_change_locations(self.DataCube,thresh=20)
        # cPickle.dump((T,I,J),open(os.path.join(TestDataDir,'AbruptTIJ.pkl'),'wb'))
        TestT,TestI,TestJ = cPickle.load(open(os.path.join(TestDataDir,'AbruptTIJ.pkl'),'rb'))
        np.testing.assert_array_equal(TestT,T)
        np.testing.assert_array_equal(TestI,I)
        np.testing.assert_array_equal(TestJ,J)
        

if __name__ == '__main__':
    unittest.main()


