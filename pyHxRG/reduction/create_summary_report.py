#!/usr/bin/env python
""" This script is to create a summary report of the images in directory."""
from __future__ import division
import sys
from astropy.io import fits
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
from astropy.visualization import HistEqStretch, ImageNormalize
import datetime
import logging

HTMLReportTemplate = """
<!DOCTYPE html>
<html>
<body>

<h1>{PageTitle}</h1>
<p> Generated on : {DateOfCreation}</p>

{Table}

</body>
</html> 
"""

TableTemplate = """
<table style="width:100%">
{0}
</table>
"""

ImageTemplate = '<img src="{0}" alt="{1}" />'

def files_in_dir(Dir,pattern='*'):
    """ Iterates over the files in Dir matching the pattern """
    for root, dirnames, filenames in os.walk(Dir):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)

def save_img(fitsfile,out_imgfilename,ext=0):
    """ Saves the fitsfile[ext] image into an image named out_imgfilename"""
    # Load img data
    Img = fits.getdata(fitsfile,ext=ext)
    # Use Histogram Equialise streaching to plot
    norm = ImageNormalize(np.nan_to_num(Img),stretch=HistEqStretch(np.nan_to_num(Img)))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(Img, origin='lower', norm=norm)
    fig.savefig(out_imgfilename)
    plt.close(fig)
    return out_imgfilename
    
def save_column_median_plot(fitsfile,ColumnMedianPlotName,ext=0):
    """ Saves the median of the column plot of fitsfile[ext] image"""
    # Load img data
    Img = fits.getdata(fitsfile,ext=ext)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.nanmedian(Img[:4,:],axis=0),color='b',label='top ref')
    ax.plot(np.nanmedian(Img[-4:,:],axis=0),color='g',label='bot ref')
    ax.plot(np.nanmedian(Img,axis=0),color='k',label='median ref')
    ax.set_xlabel('Column pixels')
    ax.set_ylabel('median e-/sec')
    ax.grid()
    ax.legend()
    fig.savefig(ColumnMedianPlotName)
    plt.close(fig)
    return ColumnMedianPlotName

def save_UTR_diagnostic_plot(fitsfile,UTR_DignoPlotName,ext=3):
    """ Saves the Diagnositc up-the-ramp average plot form ext extension """
    # Load up-the-ramp diagnostic array
    UTRarray = fits.getdata(fitsfile,ext=ext)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i,utr in enumerate(UTRarray):
        ax.plot(utr,label=str(i))
    ax.set_xlabel('No of NDRs')
    ax.set_ylabel('e-')
    ax.grid()
    ax.legend()
    fig.savefig(UTR_DignoPlotName)
    plt.close(fig)
    return UTR_DignoPlotName
    
def list_diagnostic_quantities(fitsfile):
    """ Returns a list of diagnotic quantities form the fits file """
    hdulist = fits.open(fitsfile)

    OutList = []
    
    OutList.append('No: of NDRs = {0}'.format(hdulist[0].header['NoNDR']))
    OutList.append('No: of Reset Anomaly = {0}'.format(hdulist[0].header['NRESETA']))
    OutList.append('No: of CR Hits = {0}'.format(hdulist[0].header['NOOFCRH']))
    OutList.append('No: of NaNs = {0}'.format(np.sum(np.isnan(hdulist[0].data))))
    OutList.append('95 percentile e/sec = {0}'.format(np.nanpercentile(hdulist[0].data,95)))
    OutList.append('Median varience = {0}'.format(np.nanmedian(hdulist[1].data)))
    OutList.append('Max-Min UTR derivative = {0}'.format(hdulist[3].header['MINMAXD']))
    OutList.append('STDev of UTR derivative = {0}'.format(hdulist[3].header['STD_D']))    

    hdulist.close()

    return OutList


def create_an_html_table_row(ListOfEntires,tag='td'):
    """ Returns a single table Row string with ListOfEntires """
    OutString = """
    <tr>
    {0}
    </tr>
    """.format('\n'.join(['<{0}> {1} </{0}>'.format(tag,entry) for entry in ListOfEntires]))
    return OutString

def create_an_html_list(ListOfEntires,tag='ul'):
    """ Returns an html list with ListOfEntires """
    OutString = """
    <{0}>
    {1}
    </{0}>
    """.format(tag,'\n'.join(['<li> {0} </li>'.format(entry) for entry in ListOfEntires]))
    return OutString

    
def create_summary_report(DataDir,OutReportDir):
    """ Creates a Summary of fits files in DataDir into OutReportDir """
    FigureDIR = 'figures' # Sub-directory to keep figures in
    OutFigureDir = os.path.join(OutReportDir,FigureDIR)

    HTMLDic = {'PageTitle': 'Summary of Slope Images'}

    try:
        os.makedirs(OutFigureDir)
    except OSError as e:
        logging.info(e)
        logging.info('Ignore above msg if the Output dir exists. ')
        
    TableHeader = create_an_html_table_row(['Filename','Image','Column plot','UTR plot','Quantities'],tag='th')
    TableRowsList = []
    for fitsfile in sorted(files_in_dir(DataDir,pattern='*.fits')):
        logging.info('Analysing {0}'.format(fitsfile))
        ofitsfile = fitsfile.replace(os.path.sep,'_')
        # First create a postage stamp of the image
        OutImageName = os.path.join(OutFigureDir,ofitsfile+'.png')
        OutImageName = save_img(fitsfile,OutImageName)
        oOutImageName = os.path.join(FigureDIR,os.path.split(OutImageName)[-1]) # out path for html page
        # Save the vertical median plot
        ColumnMedianPlotName = os.path.join(OutFigureDir,ofitsfile+'_Col_median.png')
        ColumnMedianPlotName = save_column_median_plot(fitsfile,ColumnMedianPlotName)
        oColumnMedianPlotName = os.path.join(FigureDIR,os.path.split(ColumnMedianPlotName)[-1])
        #Save the up-the-ramp Diagnositc plot
        UTR_DignoPlotName = os.path.join(OutFigureDir,ofitsfile+'_utr_plot.png')
        UTR_DignoPlotName = save_UTR_diagnostic_plot(fitsfile,UTR_DignoPlotName,ext=3)
        oUTR_DignoPlotName = os.path.join(FigureDIR,os.path.split(UTR_DignoPlotName)[-1])
        # List of Diagnostic Quantities
        List_of_Diagnostic_Quant = list_diagnostic_quantities(fitsfile)
        
        TableRowEntree = [fitsfile,
                          ImageTemplate.format(oOutImageName,'Slope Image'),
                          ImageTemplate.format(oColumnMedianPlotName,'Plot of median values in column'),
                          ImageTemplate.format(oUTR_DignoPlotName,'Average up-the-ramp curves of regions'),
                          create_an_html_list(List_of_Diagnostic_Quant,tag='ul')]
        TableRowsList.append(create_an_html_table_row(TableRowEntree,tag='td'))

    HTMLDic['DateOfCreation'] = datetime.datetime.utcnow()
    HTMLDic['Table'] = TableTemplate.format('\n'.join([TableHeader]+TableRowsList))

    OutputHTML = HTMLReportTemplate.format(**HTMLDic)
    
    with open(os.path.join(OutReportDir,'SlopeimageReport.html'),'w') as outfile:
        outfile.write(OutputHTML)
    
    return OutputHTML

def main():
    logging.basicConfig(level=logging.INFO)
    OutMasterReportDir = sys.argv[-1]
    for DataDir in sys.argv[1:-1]:
        _ = create_summary_report(DataDir,os.path.join(OutMasterReportDir,DataDir))


if __name__ == "__main__":
    main()
