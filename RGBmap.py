"""RGBmap module for FRESCO package.
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
from rasterio.crs import CRS

def open_raw(path_img_IF , path_hdr_IF , path_img_SR , path_hdr_SR):
    """
    Function to open the spectral parameters and spectral reflectance datacubes.
    
    Parameters
    ----------
    path_img_IF : string
        Complete path of the reflectance datacube.
    path_hdr_IF : string
        Complete path of the reflectance datacube header.
    path_img_SR : string
        Complete path of the spectral parameters datacube.
    path_hdr_SR : string
        Complete path of the spectral parameter datacube header.
        
    Returns
    -------
    img : python spectral.io.bsqfile.BsqFile object
        Spectral reflectances datacube.
    img_sr : python spectral.io.bsqfile.BsqFile object
        Spectral parameters datacube.
    wavelength : list
        List of the CRISM observation wavelengths.
    sr_names : list
        List of the CRISM spectral parameters.
    """
    
    img = envi.open(path_hdr_IF , path_img_IF)

    img_sr = envi.open(path_hdr_SR , path_img_SR)

    wavelength = np.array(img.metadata['wavelength']).astype(float)

    sr_names = img_sr.metadata['band names']

    return img , img_sr , wavelength , sr_names

class RGBImageManipulator():
    """
    Class to generate the RGB map. With this class it is possible to manually control the contrast of each RGB channel and, after the RGB map is produced, it possible to save it both in .txt and .tiff and also to georeferentiate it.
    
    Parameters
    ----------
    img : python spectral.io.bsqfile.BsqFile object
        The spectral reflectance datacube.
    img_sr : python spectral.io.bsqfile.BsqFile object
        The spectral parameter datacube.
    preset : string
        The preset as given in Viviano-Beck et al., 2014 (https://doi.org/10.1002/2014JE004627). If None then no preset is selected.
    ch1 : int
        If preset is None, this is the index of the spectral parameter used in the Red channel.
    ch2 : int
        If preset is None, this is the index of the spectral parameter used in the Green channel.
    ch3 : int
        If preset is None, this is the index of the spectral parameter used in the Blue channel.
    wavelength : list or array
        The wavelengths detected by CRISM.
    """
    def __init__(self , img , img_sr , preset , ch1 , ch2 , ch3 , wavelength):

        self.img = img
        self.img_sr = img_sr
        self.preset = preset
        self.ch1 = ch1
        self.ch2 = ch2
        self.ch3 = ch3
        self.w = wavelength

    def RGB_Viviano_Beck_2014(self,  preset):
        """
        This function uploads descriptions, spectral parameter names and combinations for RGB map as given in Viviano-Beck et al., 2014 (https://doi.org/10.1002/2014JE004627).
        Used in the function RGB_map_slider.
        
        Parameters
        ----------
        preset : string
            Pre-selected spectral parameters for RGB map generation as given in Viviano-Beck et al., 2014.
            
        Returns
        -------
        names : list
            List of the spectral parameters corresponding to the RGB channels.
        descriptions : string
            Description of the RGB map as given in Viviano-Beck et al., 2014.
        indexes : list
            Indexes of the spectral parameters for the RGB map as they sorted in the spectral parameter datacube.
        """
        sr = ['R770', 'RBR', 'BD530_2', 'SH600_2', 'SH770', 'BD640_2', 'BD860_2', 'BD920_2', 'RPEAK1',
              'BDI1000VIS', 'R440', 'IRR1', 'BDI1000IR', 'OLINDEX3', 'R1330', 'BD1300', 'LCPINDEX2', 
              'HCPINDEX2', 'VAR', 'ISLOPE1', 'BD1400', 'BD1435', 'BD1500_2', 'ICER1_2', 'BD1750_2', 
              'BD1900_2', 'BD1900R2', 'BDI2000', 'BD2100_2', 'BD2165', 'BD2190', 'MIN2200', 'BD2210_2',
              'D2200', 'BD2230', 'BD2250', 'MIN2250', 'BD2265', 'BD2290', 'D2300', 'BD2355', 'SINDEX2', 
              'ICER2_2', 'MIN2295_2480', 'MIN2345_2537', 'BD2500_2', 'BD3000', 'BD3100', 'BD3200', 
              'BD3400_2', 'CINDEX2', 'BD2600', 'IRR2', 'IRR3', 'R530', 'R600', 'R1080', 'R1506', 'R2529', 
              'R3920']

        # Descriptions from Viaviano-Beck et al. 2014

        descriptions = {'TRU': 'Enhanced true color.' , 
                        'VNA': 'Photometric correct I/F, used to correlate morphology and spectral variation.' , 
                        'FEM': 'Fe minerals absorption.' , 
                        'FM2': 'Complementary info on Fe minerals.' , 
                        'FAL': 'False color image. Red/orange -> olivine rich , blue/green -> clay , green -> carbonates , gray/brown -> basaltic .' ,
                        'MAF': 'Mafic mineralogy. Green/cyan -> Low Ca Pyroxene , blue/magenta -> High Ca Pyroxene.' , 
                        'HYD': 'Hydrated mineralogy. Magenta -> polyhydrated sulfates , yellow/green -> monohydrated sulfates , blue -> hydrated minerals .' ,
                        'PHY': 'Phyllosilicates.  Red -> non-hydr. Fe/Mg-OH minerals , magenta -> hydr. Fe/Mg-OH inerals , green -> non-hydr. Al/Si-OH minerals , cyan -> hydr. Al/Si-OH minerals , blue -> other hydrated minerals.' , 
                        'PFM': 'Phyllosilicates with Fe and Mg. Red/yellow -> prehnite, chlorite, epidote or Ca/Fe carbonate, cyan -> Fe/Mg smectites of Mg carbonates.' , 
                        'PAL': 'Phyllosilicates qith Al. Red/yellow -> Al smectites or hydrated silica, cyan -> alunite, light/white -> kaolinite.' ,
                        'HYS': 'Hydrated silica. Used to differentiate between Al-phyl and hyd. silica. Light red/yellow -> hydrated silica, yellow -> jarosite,  cyan -> Al-OH minerals, blue -> sulfates, clays, hydr. silica, carbonates or water ice.' ,
                        'ICE': 'H20/CO2 ice. Red -> sulfates, clays, hydr. silica, carbonate, water ice. Green -> Water ice, Blue -> carbon dioxide ice.' ,
                        'IC2': 'Complementary information about H20/CO2 ice. Red -> ice free surface, green -> water ice, blue -> carbon dioxide ice.' ,
                        'CHL': 'Info about chloride deposits. Yellow/green -> hydr. minerals and phyllosilicates, blue -> chloride.' ,
                        'CAR': 'Info about Mg carbonate minerals. Red/magenta -> Fe/Mg phyllosilicates, yellowish-white-bluish -> Mg carbonates, blue -> sulfates, clays, hydr. silica or carbonate.' ,
                        'CR2': 'Info to distinguish carbonate minerals. Red/magenta -> Mg-carbonates, green/cyan -> Fe/Ca carbonates.'
                       }

        names = {'TRU' : [sr.index('R600') , sr.index('R530') , sr.index('R440')] , 
                 'VNA' : [sr.index('R770') , sr.index('R770') , sr.index('R770')] , 
                 'FEM' : [sr.index('BD530_2') , sr.index('SH600_2') , sr.index('BDI1000VIS')] , 
                 'FM2' : [sr.index('BD530_2') , sr.index('BD920_2') , sr.index('BDI1000VIS')] , 
                 'FAL' : [sr.index('R2529') , sr.index('R1506') , sr.index('R1080')] , 
                 'MAF' : [sr.index('OLINDEX3') , sr.index('LCPINDEX2') , sr.index('HCPINDEX2')] , 
                 'HYD' : [sr.index('SINDEX2') , sr.index('BD2100_2') , sr.index('BD1900_2')] , 
                 'PHY' : [sr.index('D2300') , sr.index('D2200') , sr.index('BD1900R2')] , 
                 'PFM' : [sr.index('BD2355') , sr.index('D2300') , sr.index('BD2290')] , 
                 'PAL' : [sr.index('BD2210_2') , sr.index('BD2190') , sr.index('BD2165')] ,
                 'HYS' : [sr.index('MIN2250') , sr.index('BD2250') , sr.index('BD1900R2')] ,
                 'ICE' : [sr.index('BD1900_2') , sr.index('BD1500_2') , sr.index('BD1435')] ,
                 'IC2' : [sr.index('R3920') , sr.index('BD1500_2') , sr.index('BD1435')] ,
                 'CHL' : [sr.index('ISLOPE1') , sr.index('BD3000') , sr.index('IRR2')] , 
                 'CAR' : [sr.index('D2300') , sr.index('BD2500_2') , sr.index('BD1900_2')] ,
                 'CR2' : [sr.index('MIN2295_2480') , sr.index('MIN2345_2537') , sr.index('CINDEX2')]
                }
        
        print( names[self.preset] , [ sr[names[self.preset][0]] , sr[names[self.preset][1]] , sr[names[self.preset][2]] ])
        return names[self.preset] , descriptions[self.preset] , [ sr[names[self.preset][0]] , sr[names[self.preset][1]] , sr[names[self.preset][2]] ]

    def f(self, RGB, min_R, min_G, min_B, max_R, max_G, max_B, clip=False):
        """
        This function uploads the RGB map during the customization. This function is only used inside RGB_map_slider.
        
        Parameters
        ----------
        RGB : 3-dim array
            The RGB image to be updated.
        min_R : float
            Minimum value for the Red channel.
        min_G : float
            Minimum value for the Green channel.
        min_B : float
            Minimum value for the Blue channel.
        max_R : float
            Maximum value for the Red channel.
        max_G : float
            Maximum value for the Green channel.
        max_B : float
            Maximum value for the Blue channel.
        clip : bool
            If True it clips the negative values. Default if False.
            
        Returns
        -------
        RGB_raw : 3-dim array
            Updated RGB map
        """
        stretches_min = [min_R , min_G , min_B]
        stretches_max = [max_R , max_G , max_B]

        stretch = np.array([ [min_R , max_R] , [min_G , max_G] , [min_B , max_B] ])

        RGBs = np.where(RGB < stretch[:,0], stretches_min, RGB) 
        RGBs = np.where(RGB > stretch[:,1], stretches_max, RGB) 

        RGB_raw = (RGBs - stretch[:,0]) / (stretch[:,1] - stretch[:,0])

        if clip == True:

            RGB_raw = np.clip(RGB_raw , 0. , 1.)

        return RGB_raw

    def area(self, hist, bins, line1, line2):
        """
        Function to calulate the percentile area of a histogram between two lines. Only used inside RGB_map_slider.
        
        Parameters
        ----------
        hist : array
            Histogram of the value of the RGB channel.
        bins : int
            Number of bins of the histogram.
        line1 : matplotlib.axes.Axes
            The line object of the left percentiles.
        line2 : matplotlib.axes.Axes
            The line object of the right percentiles.
            
        Returns
        -------
        Areas : list
            List containing the area of the histogram on the left of the first line and on the right of the second line.
        """
        area1 , area2 , total_area = 0 , 0 , 0
        for S in range(len(bins)-1):

            F = S + 1

            total_area += hist[S]*(bins[F] - bins[S])

            if bins[S] <= line1.get_xdata():
                area1 += hist[S]*(bins[F] - bins[S])
            if bins[S] <= line2.get_xdata():
                area2 += hist[S]*(bins[F] - bins[S])

        area_inferior = area1*100/total_area
        area_superior = area2*100/total_area

        return [area_inferior , area_superior]

    def Labels(self):
        L = {'TRU': [['--' , 'white' , 'white'] , ['--' , 'white' , 'white']],#'Enhanced true color.' , 
          'VNA': [['--' , 'white' , 'white'] , ['--' , 'white' , 'white']],#'Photometric correct I/F, used to correlate morphology and spectral variation.' , 
          'FEM': [['--' , 'white' , 'white'] , ['--' , 'white' , 'white']],#'Fe minerals absorption.' , 
          'FM2': [['--' , 'white' , 'white'] , ['--' , 'white' , 'white']],#'Complementary info on Fe minerals.' , 
          'FAL': [['Olivine' , 'red' , 'orange'] , ['Clay' , 'mediumseagreen' , 'blue'] , ['Carbonates' , 'green' , 'green'] , ['Basalts' , 'gray' , 'brown']] ,
          'MAF': [['Olivine' , 'red' , 'red'] , ['Low Ca Pyroxene' , 'green' , 'cyan'] , ['High Ca Pyroxene' , 'blue' , 'magenta']],
          'HYD': [['Polyhydrated sulfates' , 'magenta' , 'magenta'] , ['Monohydrated sulfates' , 'yellow' , 'green'] , ['Hydrated minerals' , 'blue' , 'blue']],
          'PHY': [['Non hydrated Fe/Mg-OH' , 'red' , 'red'] , ['Hydrated Fe/Mg-OH' , 'magenta' , 'magenta'] , ['Non hydrated Al/Si-OH' , 'green' , 'green'] , ['Hydrated Al/Si-OH' , 'cyan' , 'cyan'] , ['Hydrated minerals' , 'blue' , 'blue']], 
          'PFM': [['Prehnite' , 'red' , 'yellow'] , ['Chlorite' ,  'red' , 'yellow'] , ['Epidote' , 'red' , 'yellow'] , ['Ca/Fe carbonate', 'red' , 'yellow'] , ['Fe/Mg smectites / Mg carbonates' , 'cyan' , 'cyan'] , ['Kaolinite' , 'white' , 'white']] ,
          'PAL': [['Al smectites/Hydrated silica' , 'red' , 'yellow'] , ['Alunite' , 'cyan' , 'cyan'] , ['Kaolinite' , 'white' , 'white']] ,
          'HYS': [['Hydrated silica' , 'red' , 'yellow'] , ['Jarosite' , 'yellow' , 'yellow'] , ['Al-OH minerals' , 'cyan' , 'cyan'] , ['Other hydrates' , 'blue' , 'blue']] ,
          'ICE': [['Other hydrates' , 'red' , 'red'] , ['H2O ice' , 'green' , 'green'] , ['CO2 ice' , 'blue' , 'blue'] ] ,
          'IC2': [['Ice free surface' , 'red' ,'red'] , ['H2O ice' , 'green' , 'green'] , ['CO2 ice' , 'blue' , 'blue'] ] , 
          'CHL': [['Hydr. mineral and phyllosilicates' , 'yellow' , 'green'] , ['Chloride' , 'blue' , 'blue']],
          'CAR': [['Fe/Mg phyllosilicates' , 'red' , 'magenta'] , ['Mg carbonates' , 'yellow' , 'lightblue'] , ['Other hydrates' , 'blue' , 'blue'] ] ,
          'CR2': [['Mg carbonates' , 'red' , 'magenta'] , ['Fe/Ca carbonates' , 'green' , 'cyan']]
         }
        return L

    def RGBmapmake(self, FALSE , bi ,clip , cumhist ,preset_true_colors , use_false_color ,
                       R_min_in = [0,1]  , R_max_in = [0,1]  ,
                       G_min_in = [0,1]  , G_max_in = [0,1]  ,
                       B_min_in = [0,1]  , B_max_in = [0,1]  ,
                       init_R = [0,1]  , init_G = [0,1]  , init_B = [0,1]  ,
                       slider_step = 0.005 ,
                       slider_height = 0.02 , slider_width = 0.25 , slider_spacing = 0.05):
        """
        Function to perform the customization of the RGB map by moving apposite sliders to enhance constrast between different colors (i.e. spectral parameters).
        To finish the customization it is sufficent to close the plot window.
        
        Parameters
        ----------
        FALSE : 3-dim array
            RGB image to be used as background.
        bi : int
            Number of bins to divide the histograms into.
        clip : bool
            If to clip the negative values or not.
        cumhist : bool
            If to use cumulative histograms instead of frequency histograms.
        preset_true_colors : string
            If use_false_color is False, here insert the preset name from Viviano-Beck et al., 2014 that wants to be used as background true color RGB image.
        use_false_color : bool
            If to use a pre-computed RGB background map or to select another one from Viviano-Beck et al., 2014 as it is (without stretching).
        R_min_in : list of two floats
            Minimum and maximum possible values for the minimum of the Red channel. Default is [0,1].
        R_max_in : list of two floats
            Minimum and maximum possible values for the maximum of the Red channel. Default is [0,1].
        G_min_in : list of two floats
            Minimum and maximum possible values for the minimum of the Green channel. Default is [0,1].
        G_max_in : list of two floats
            Minimum and maximum possible values for the maximum of the Green channel. Default is [0,1].
        B_min_in : list of two floats
            Minimum and maximum possible values for the minimum of the Blue channel. Default is [0,1].
        B_max_in : list of two floats
            Minimum and maximum possible values for the maximum of the Blue channel. Default is [0,1].
        init_R : list of two floats
            Initial values of the Red channel. Default is [0,1].
        init_G : list of two floats
            Initial values of the Green channel. Default is [0,1].
        init_B : list of two floats
            Initial values of the Blue channel. Default is [0,1].
        slider_step : float
            Minimum step done by the slider while interacting with it. Default is 0.005.
        slider_height : float
            Height at which the sliders are posed in the plot. Default is 0.02.
        slider_width : float
            Width of the sliders. Default is 0.25.
        slider_spacing : float
            Space between the sliders. Default is 0.05.
            
        Returns
        -------
        RGB : 3-dim array
            Final RGB map in the form of a numpy array.
        stretches : list of float
            Final stretch values in the following order: final_min_R , final_min_G , final_min_B , final_max_R , final_max_G , final_max_B.
        """
        fig , ax = plt.subplots(1 , 4 , figsize = [10,5] , gridspec_kw={'width_ratios': [1,1,1,3]})
    
        # setting initial stretch to float
        init_min_R, init_min_G, init_min_B = float(init_R[0]) , float(init_G[0]) , float(init_B[0])
        init_max_R, init_max_G, init_max_B = float(init_R[1]) , float(init_G[1]) , float(init_B[1])

        # Creation of the RGB image either from coded ones or from custom select indices
        if self.preset == None:

            sr = ['R770', 'RBR', 'BD530_2', 'SH600_2', 'SH770', 'BD640_2', 'BD860_2', 'BD920_2', 'RPEAK1',
                  'BDI1000VIS', 'R440', 'IRR1', 'BDI1000IR', 'OLINDEX3', 'R1330', 'BD1300', 'LCPINDEX2', 
                  'HCPINDEX2', 'VAR', 'ISLOPE1', 'BD1400', 'BD1435', 'BD1500_2', 'ICER1_2', 'BD1750_2', 
                  'BD1900_2', 'BD1900R2', 'BDI2000', 'BD2100_2', 'BD2165', 'BD2190', 'MIN2200', 'BD2210_2',
                  'D2200', 'BD2230', 'BD2250', 'MIN2250', 'BD2265', 'BD2290', 'D2300', 'BD2355', 'SINDEX2', 
                  'ICER2_2', 'MIN2295_2480', 'MIN2345_2537', 'BD2500_2', 'BD3000', 'BD3100', 'BD3200', 
                  'BD3400_2', 'CINDEX2', 'BD2600', 'IRR2', 'IRR3', 'R530', 'R600', 'R1080', 'R1506', 'R2529', 
                  'R3920']

            sr_channels_number = [self.ch1 , self.ch2 , self.ch3]
            RGB_raw = np.array(self.img_sr[:,:,sr_channels_number].squeeze()).astype(float)
            
            ax[0].set_xlabel(sr[self.ch1] , color = 'red' , fontsize = 15)
            ax[1].set_xlabel(sr[self.ch2] , color = 'green' , fontsize = 15)
            ax[2].set_xlabel(sr[self.ch3] , color = 'blue' , fontsize = 15)

        else:
            sr_channels_number, descr , pars = self.RGB_Viviano_Beck_2014(self.preset)
            print(sr_channels_number , pars) 
            RGB_raw = np.array(self.img_sr[:,:,sr_channels_number].squeeze())

            ax[0].set_xlabel(str(pars[0]) , color = 'red' , fontsize = 15)
            ax[1].set_xlabel(str(pars[1]) , color = 'green' , fontsize = 15)
            ax[2].set_xlabel(str(pars[2]) , color = 'blue' , fontsize = 15)

        if use_false_color == False:

            # Create the true or false color image
            true_channels , true_descr , true_pars = self.RGB_Viviano_Beck_2014(preset_true_colors)
            RGB_true_colors = np.array(self.img_sr[:,:,true_channels].squeeze())
            RGB_true_colors[RGB_true_colors > 1.] = np.nan

        else:

            RGB_true_colors = FALSE

        # Remove of the borders and definition of image in function of the stretches
        RGB_raw[RGB_raw > 1.] = np.nan
        RGB_browse_norm = self.f(RGB_raw, init_R[0] , init_G[0] , init_B[0] ,
                                 init_R[1] , init_G[1] , init_B[1] , clip = clip)

        # Choosing between cumulative histogram and frequency histogram
        if cumhist == True:
            hR , biR , _ = ax[0].hist( RGB_raw[:,:,0].ravel() , bi , color = 'r' , alpha = 0.5 , density = True , cumulative = True )
            hG , biG , _ = ax[1].hist( RGB_raw[:,:,1].ravel() , bi , color = 'g' , alpha = 0.5 , density = True , cumulative = True )
            hB , biB , _ = ax[2].hist( RGB_raw[:,:,2].ravel() , bi , color = 'b' , alpha = 0.5 , density = True , cumulative = True )
        else:
            hR , biR , _ = ax[0].hist( RGB_raw[:,:,0].ravel() , bi , color = 'r' , alpha = 0.5 , density = True , cumulative = False )
            hG , biG , _ = ax[1].hist( RGB_raw[:,:,1].ravel() , bi , color = 'g' , alpha = 0.5 , density = True , cumulative = False )
            hB , biB , _ = ax[2].hist( RGB_raw[:,:,2].ravel() , bi , color = 'b' , alpha = 0.5 , density = True , cumulative = False )

        # Setting histogram graphs limits:
        ax[0].set_xlim(biR[np.argmin(biR)+1] , np.max(biR))
        ax[1].set_xlim(biG[np.argmin(biG)+1] , np.max(biG))
        ax[2].set_xlim(biB[np.argmin(biB)+1] , np.max(biB))
        ax[0].set_ylim(0 , np.max(np.sort(hR)[0:len(hR)-1]))
        ax[1].set_ylim(0 , np.max(np.sort(hG)[0:len(hG)-1]))
        ax[2].set_ylim(0 , np.max(np.sort(hB)[0:len(hB)-1]))

        # Definition of global variables
        final_min_R , final_min_G , final_min_B = None , None , None
        final_max_R , final_max_G , final_max_B = None , None , None
        RGB_final = None

        im = ax[3].imshow(RGB_browse_norm)

        im.set_data(RGB_browse_norm)

        # Plot of the vertical lines on the histograms and sacing their data
        vr = ax[0].axvline( init_R[0] , color = 'r' , linestyle = '--' , linewidth = 0.6 )
        vR = ax[0].axvline( init_R[1] , color = 'r' , linestyle = '--' , linewidth = 0.6 )
        vg = ax[1].axvline( init_G[0] , color = 'g' , linestyle = '--' , linewidth = 0.6 )
        vG = ax[1].axvline( init_G[1] , color = 'g' , linestyle = '--' , linewidth = 0.6 )
        vb = ax[2].axvline( init_B[0] , color = 'b' , linestyle = '--' , linewidth = 0.6 )
        vB = ax[2].axvline( init_B[1] , color = 'b' , linestyle = '--' , linewidth = 0.6 )

        vr.set_xdata([init_R[0]])
        vR.set_xdata([init_R[1]])
        vg.set_xdata([init_G[0]])
        vG.set_xdata([init_G[1]])
        vb.set_xdata([init_B[0]])
        vB.set_xdata([init_B[1]])

        # Percentile calculations
        areaR , areaG , areaB = self.area(hR , biR , vr , vR) , self.area(hG , biG , vg , vG) , self.area(hB , biB , vb , vB)

        aRmin , aRmax = areaR[0] , areaR[1]
        aGmin , aGmax = areaG[0] , areaG[1]
        aBmin , aBmax = areaB[0] , areaB[1]

        # Writing percentiles and saving the value
        textR = ax[0].set_title(f"{aRmin:.2f}%" + " - " + f"{aRmax:.2f}%" , color = 'r' , fontsize = 8)
        textG = ax[1].set_title(f"{aGmin:.2f}%" + " - " + f"{aGmax:.2f}%" , color = 'g' , fontsize = 8)
        textB = ax[2].set_title(f"{aGmin:.2f}%" + " - " + f"{aRmax:.2f}%" , color = 'b' , fontsize = 8)

        textR.set_text(f"Perc.: {aRmax:.2f}%")
        textG.set_text(f"Perc.: {aGmax:.2f}%")
        textB.set_text(f"Perc.: {aBmax:.2f}%")

        # Red channel sliders
        ax_min_R = plt.axes([0.05, 0.9, slider_width, slider_height])
        ax_max_R = plt.axes([0.05, 0.9 + slider_spacing, slider_width, slider_height])

        min_R_slider = Slider(ax_min_R, label = 'min R',
                              valmin = R_min_in[0], valmax = R_min_in[1],
                              valinit = init_min_R , valstep = slider_step , color = 'r')

        max_R_slider = Slider(ax_max_R, label = 'max R',
                              valmin = R_max_in[0], valmax = R_max_in[1],
                              valinit = init_max_R , valstep = slider_step , color = 'r')

        # Green channel sliders
        ax_min_G = plt.axes([0.37, 0.9, slider_width, slider_height])
        ax_max_G = plt.axes([0.37, 0.9 + slider_spacing, slider_width, slider_height])

        min_G_slider = Slider(ax_min_G, label = 'min G',
                              valmin = G_min_in[0], valmax = G_min_in[1],
                              valinit = init_min_G , valstep = slider_step , color = 'g')

        max_G_slider = Slider(ax_max_G, label = 'max G',
                              valmin = G_max_in[0], valmax = G_max_in[1],
                              valinit = init_max_G , valstep = slider_step , color = 'g')

        # Blue channel sliders
        ax_min_B = plt.axes([0.69, 0.9, slider_width, slider_height])
        ax_max_B = plt.axes([0.69, 0.9 + slider_spacing, slider_width, slider_height])

        min_B_slider = Slider(ax_min_B, label = 'min B',
                              valmin = B_min_in[0] , valmax = B_min_in[1],
                              valinit = init_min_B , valstep = slider_step , color = 'b')

        max_B_slider = Slider(ax_max_B, label = 'max B',
                              valmin = B_max_in[0], valmax = B_max_in[1],
                              valinit = init_max_B , valstep = slider_step , color = 'b')

        def on_button_clicked(event):
            nonlocal is_toggled
            is_toggled = not is_toggled

        # Define a function to update the image displayed in the plot
        is_toggled = True # Varibale to see the state of the button
        def update_image(event):
            nonlocal is_toggled, RGB_final, final_min_R, final_min_G, final_min_B, final_max_R, final_max_G, final_max_B
            if not is_toggled:
                # Change the image to a new state
                img_new = RGB_true_colors # define the new image here
                im.set_data(img_new)
                fig.canvas.draw_idle()
            else:
                # Retrieve current slider values
                min_R, max_R = min_R_slider.val, max_R_slider.val
                min_G, max_G = min_G_slider.val, max_G_slider.val
                min_B, max_B = min_B_slider.val, max_B_slider.val

                # Update final slider values
                final_min_R, final_min_G, final_min_B = min_R, min_G, min_B
                final_max_R, final_max_G, final_max_B = max_R, max_G, max_B

                # Calculate updated RGB image
                RGB_browse_norm = self.f(RGB_raw, min_R, min_G, min_B, max_R, max_G, max_B, clip)
                im.set_data(RGB_browse_norm)

                RGB_final = RGB_browse_norm

                # Calculate updated vertical lines on histograms
                vr.set_xdata([min_R])
                vR.set_xdata([max_R])
                vg.set_xdata([min_G])
                vG.set_xdata([max_G])
                vb.set_xdata([min_B])
                vB.set_xdata([max_B])

                # Calculate updated percentiles
                areaR , areaG , areaB = self.area(hR , biR , vr , vR) , self.area(hG , biG , vg , vG) , self.area(hB , biB , vb , vB)

                aRmin , aRmax = areaR[0] , areaR[1]
                aGmin , aGmax = areaG[0] , areaG[1]
                aBmin , aBmax = areaB[0] , areaB[1]

                textR.set_text(f"{aRmin:.2f}%" + " - " + f"{aRmax:.2f}%")
                textG.set_text(f"{aGmin:.2f}%" + " - " + f"{aGmax:.2f}%")
                textB.set_text(f"{aBmin:.2f}%" + " - " + f"{aBmax:.2f}%")

                # Change the image back to the original state
                fig.canvas.draw_idle()

        # Create button
        button_ax = plt.axes([0.8, 0.82, 0.1, 0.07])
        button = Button(button_ax, 'Change image')
        button.on_clicked(on_button_clicked)
        button.on_clicked(update_image)

        # Set up update function to be called when sliders are changed
        for slider in [min_R_slider, max_R_slider, min_G_slider, max_G_slider, min_B_slider, max_B_slider]:
            slider.on_changed(update_image)

        # Set up title
        if self.preset != None:
            ax[3].set_title(self.preset)
            if self.preset != 'TRU' and self.preset != 'VNA' and self.preset != 'FEM' and self.preset != 'FM2':
                
                patches = []
                labe = []

                L = self.Labels()[self.preset]

                for i in range(len(L)):

                    lab , c1 , c2 = L[i][0] , L[i][1] , L[i][2]

                    labe.append(str(' ') + lab)

                    m1, = plt.plot([], [], c=c1 , marker='s', markersize=20,
                                  fillstyle='left', linestyle='none')

                    m2, = plt.plot([], [], c=c2 , marker='s', markersize=20,
                                  fillstyle='right', linestyle='none')

                    patches.append((m1,m2))

                plt.legend(( patches ), (labe), numpoints = 1, labelspacing = 2, ncol = 2 , frameon=False ,
                          handletextpad = 1, handlelength = 1.5 , columnspacing = 5 ,
                          loc = 'lower right', fontsize = 10 , bbox_to_anchor = (1,-11) )

            
            else:
                print('No legend')
        else:
            ax[3].set_title('Custom map')
        ax[3].axis('off')

        # Creating mask to set NaN values outside the images to be shown in white instead of black
        MASK = np.isnan(RGB_browse_norm[:,:,0])
        ALPHA = np.zeros((RGB_browse_norm.shape[0] , RGB_browse_norm.shape[1]))
        ALPHA[MASK] = 1
        ax[3].imshow(ALPHA , alpha = ALPHA , cmap = 'Greys_r')

        plt.show()
        
        self.RGB_final = RGB_final

        # Return the final map
        return self.RGB_final , [final_min_R , final_min_G , final_min_B , final_max_R , final_max_G , final_max_B]
    
    def savemap(self , name , folder = None , extension = None , show = False):
        """
        Function to save the RGB map into 3 separated .txt files (one for each channel) and, if given, also in a specific image format.
        
        Parameters
        ----------
        name : string
            Name with which the RGB map is saved.
        folder : string
            Path of the folder in which the RGB map is saved. If None it saves in home directory. Default is None.
        extension : string, '.tif' or '.png' or '.jpg' or '.pdf'
            Extension in which to save the RGB map as an image. If None it does not save it as an image. Default is None.
        show : bool
            To show or not the resulting RGB map saved as an image. Deafult is None.
            
        Returns
        -------
        None
        """

        R = np.zeros((self.RGB_final.shape[0] , self.RGB_final.shape[1]))
        G = np.zeros((self.RGB_final.shape[0] , self.RGB_final.shape[1]))
        B = np.zeros((self.RGB_final.shape[0] , self.RGB_final.shape[1]))

        for i in range(self.RGB_final.shape[0]):
            for j in range(self.RGB_final.shape[1]):
                R[i,j] = self.RGB_final[i,j,0]
                G[i,j] = self.RGB_final[i,j,1]
                B[i,j] = self.RGB_final[i,j,2]

        if folder == None:
            np.savetxt(name + '_R.txt' , R)
            np.savetxt(name + '_G.txt' , G)
            np.savetxt(name + '_B.txt' , B)
        else:
            np.savetxt(folder + name + '_R.txt' , R)
            np.savetxt(folder + name + '_G.txt' , G)
            np.savetxt(folder + name + '_B.txt' , B)
            
        if extension == '.tif' or extension == '.png' or extension == '.jpg' or extension == '.pdf' or extension == '.svg':
            
            #IMG = self.RGB_final.resize( (self.img.shape[1] , self.img.shape[0] , 3) )

            #self.RGB_final.save(folder + name + extension , bbox_inches = 'tight' , pad_inches = 0 , transparent = True)
            plt.imshow(self.RGB_final)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(folder + name + extension , bbox_inches = 'tight' , pad_inches = 0 , transparent = True)
            if show == True:
                plt.show()
            else:
                plt.close()
                
        return 
    
    def georeference(self , wkt , name , folder , min_lat , max_lat , westernmost_lon , easternmost_lon):
        '''
        Function to georeference a .tif image to a given reference frame.
        
        Parameters
        ----------
        wkt : string
            Reference system in which the image has to be georeferenced.  
            Example for Mars --> wkt = "GEOGCRS[\"GCS_Mars_2000\",DATUM[\"D_Mars_2000\",
                                           ELLIPSOID[\"Mars_2000_IAU_IAG\",3396190,169.894447223612,
                                           LENGTHUNIT[\"metre\",1]]],PRIMEM[\"Reference_Meridian\",0,
                                           ANGLEUNIT[\"degree\",0.0174532925199433]],CS[ellipsoidal,2],
                                           AXIS[\"geodetic latitude (Lat)\",north,ORDER[1],
                                           ANGLEUNIT[\"degree\",0.0174532925199433]],
                                           AXIS[\"geodetic longitude (Lon)\",east,ORDER[2],
                                           ANGLEUNIT[\"degree\",0.0174532925199433]],
                                           USAGE[SCOPE[\"unknown\"],AREA[\"World\"],
                                           BBOX[-90,-180,90,180]],ID[\"ESRI\",104905]]"
        name : string
            Name of the .tif image you want to georeference.
        fodler : string
            Path of the image you want to save.
        min_lat : float
            Southernmost latitude of the image.
        max_lat : float
            Northernmost latitude of the image.
        westernmost_lon : float
            Westernmost longitude of the image.
        Easternmost_lon : float
            Easternmost longitude of the image.
            
        Returns
        -------
        None
        '''

        height = self.img_sr.shape[0] #lines
        width = self.img_sr.shape[1] #samples

        tl = GroundControlPoint(0, 0, westernmost_lon, max_lat) #top left 
        bl = GroundControlPoint(height, 0, westernmost_lon, min_lat) #bottom left 
        br = GroundControlPoint(height, width, easternmost_lon, min_lat) #bottom right
        tr = GroundControlPoint(0, width, easternmost_lon, max_lat) #top right

        gcps = [tl, bl, br, tr]

        transform = from_gcps(gcps)
        crs = CRS.from_wkt(wkt)

        with rasterio.open(folder + name + '.tif', 'r+') as ds:
            ds.crs = crs
            ds.transform = transform
            
