"""
    SpectraAnalysis module
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.widgets import PolygonSelector
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseButton
from matplotlib.ticker import FuncFormatter

import numpy as np
import pandas as pd

from scipy.stats import median_abs_deviation as MAD
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.signal import savgol_filter, argrelextrema, find_peaks

from kneed import KneeLocator

import torch
import torch.nn.functional as F
from torch import nn
class SpectraAnalysis():
    """
    Class to perform the required spectral analyses. 
    
    Parameters
    ----------
    final : 1-dim array
        Normalized spectrum as given by SpectraNorm.norm_spectra().
    m_spec : 1-dim array
        Mean target spectrum.
    err_spec : 1-dim array
        Standard deviation of the target spectrum.
    n_spectra : 1-dim array
        Median neutral spectrum.
    n_err : 1-dim array
        MAD of the neutral spectrum.
    wavelength : list or array
        CRISM observation wavelengths.
    MIN : float
        Minimum wavelength value to be taken into consideration.
    MAX : float
        Maximum wavelength value to be taken into consideration.
    folder : string
        Folder in which the data spectra_MICA_LAB_info.csv is stored. Default is pyFRESCO/data.
    """
    def __init__(self , final , final_error , m_spec , err_spec , n_spectra , n_err , wavelength , MIN , MAX , folder = 'pyfresco/data'):
        
        self.final = final # normalized spectrum
        self.final_error = final_error # normalized spectrum propagated error
        self.err_spec = err_spec # error of the target
        self.m_spec = m_spec # target mean spectrum
        self.n_spectra = n_spectra # neutral median spectrum
        self.n_err = err_spec # neutral error
        self.w = wavelength # wavelength
        self.MIN = MIN # minimum wavelength value
        self.MAX = MAX # maximum wavelength value
        self.folder = folder
        
    def limits(self, other_w = None):
        """
        Function to compute the limits of the x-axis and the corresponding indexes to then compute the y-axis limits of an array given the list/array of the x axis.
            
        Parameters
        ----------
        other_w : list, array or None
            If list or array, this will be taken as the wavelength array to use, if None then the CRISM observation wavelengths are used. Default is None.
        
        Returns
        -------
        extremas : list of float
            list of index correspondant do xmin , index correpsondant to xmax, xmin and xmax
        """
        
        if type(other_w) == list or isinstance(other_w , np.ndarray) == True:
            k = other_w
        else:
            k = self.w
        
        a , b = np.zeros(len(k)) , np.zeros(len(k))
        for i in range(len(k)):
            a[i] = np.abs(k[i]-self.MIN)
            b[i] = np.abs(k[i]-self.MAX)
        xmin_ind , xmax_ind = np.argmin(a) , np.argmin(b)
        xmin , xmax = k[xmin_ind] , k[xmax_ind]

        return [xmin_ind , xmax_ind , xmin , xmax]
    
    def find_nearest(self, array, value):
        """
        Function used to find the index of element nearest to a given arbitrary value.
        
        Parameters
        ----------
        array : array
            Array in which to search
        value : float
            Value to search the nearest element inside array.
            
        Returns
        -------
        idx : int
            Index of the nearest value.
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def upload_norm(self , name , folder):
        """
        Function to upload a pre-made median/mad spectrum.
        
        Parameters
        ----------
        name : string
            Name of the file.
        folder : string or None
            If None it will be saved into the home folder, if a folder path is given, the path must end with the /.
        
        Returns
        -------
        norm : 1d-array
            The uploaded median spectrum.
        w : 1d-array
            The cut wavelength range.
        """
        
        i, j = self.find_nearest(self.w, self.MIN), self.find_nearest(self.w, self.MAX)

        if folder == None:
            self.w = np.genfromtxt('Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            self.final = np.genfromtxt(name + '.txt')
        else:
            self.w = np.genfromtxt(folder + 'Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            self.final = np.genfromtxt(folder + name + '.txt')
        
        return self.final , self.w
    
    def moving_average(self , window_size , limiti = True):
        """
        Function to perform a moving average smoothing on the normalized spectrum.
        
        Parameters
        ----------
        window_size : int
            Size of the step taken for the moving mean.
        
        Returns
        -------
        result : 1-dim array
            Moving-mean smoothed normalized spectrum.
        """
        if window_size < 1:
            raise ValueError("Window_size must be at least 1.")

        data = np.asarray(self.final)
        result = np.empty(len(self.final))
        half_window = window_size // 2

        for i in range(len(self.final)):
            start = max(0, i - half_window)
            end = min(len(self.final), i + half_window + 1)
            result[i] = np.mean(data[start:end])
        self.final_smooth = result

        if limiti == True:
            lims = self.limits()
            a , b = lims[0] , lims[1]
            plt.plot(self.w[a:b] , self.final , 'b')
            plt.plot(self.w[a:b] , self.final_smooth , 'r')
        else:
            plt.plot(self.w , self.final , 'b')
            plt.plot(self.w , self.final_smooth , 'r')
        plt.xlabel('$\lambda$[nm]')
        plt.show()
        
        return result
    
    def savgol(self, window, order , limiti = True):
        """
        Function to perform a Savitzky-Golay smoothing on the normalized spectrum as given in https://pubs.acs.org/doi/10.1021/ac60214a047.
        Mutuated from scipy.signal (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html).
        
        Parameters
        ----------
        window : int
            Size of the convolution window.
        order : int
            Order of the convolutional fit.
        
        Returns
        -------
        result : 1-dim array
            Moving-mean smoothed normalized spectrum.
        """
        
        self.final_smooth = savgol_filter(self.final , window , order)
        self.final_smooth_error = np.zeros(len(self.final_smooth))

        if limiti == True:
            lims = self.limits()
            a , b = lims[0] , lims[1]
            plt.plot(self.w[a:b] , self.final , 'b')
            plt.plot(self.w[a:b] , self.final_smooth , 'r')
        else:
            plt.plot(self.w , self.final , 'b')
            plt.plot(self.w , self.final_smooth , 'r')
        plt.xlabel('$\lambda$[nm]')
        plt.show()
        return self.final_smooth

    def dataset_interaction(self, mineralname, use):
        """
        Utils function to interact with the .csv file named "spectra_MICA_LAB_info.csv" to extract the needed spectral informations from the comparison with the references given by the MICA files, 2019 (http://crism.jhuapl.edu/data/mica/).
        IMPORTANT: THE MICA FILES REFERENCE LABORATORY SPECTRA ARE IN https://crismtypespectra.rsl.wustl.edu/ AND FOR SOME REASONS SOME SPECTRA RESULTS UNAVALIABLE FROM THERE, BEING: CO2 ICE, HYDRATED SILICA, HYDROXYLATED FE SULFATE, GYPSUM AND CHLORIDE. THUS, REGARDING THESE MINERALS ONLY THE CRISM REFERENCE IS USED AT THE MOMENT.

        Parameters
        ----------
        mineralname : string
            Name of the mineral from the MICA files.
        use : string
            If to use only the CRISM certified spectra, the laboratory spectra or both of them. Accepted values are either MICA, LAB or Both.
        
        Returns
        -------
        index : int
            Index of the given mineral name in the spectra_MICA_LAB_info.csv .
        """
        MICA_spectra_names = ['Hematite' , 'Mg Olivine' , 'Fe Olivine' , 'Plagioclase' , 'Low Ca Pyroxene' , 'High Ca Pyroxene' , 
                              'H20 Ice' , 'CO2 Ice' , 
                              'Monohydrated Sulfate' , 'Alunite' , 'Hydroxylated Fe-Sulfate' , 'Jarosite' , 'Polyhydrated Sulfate' , 'Gypsum' , 'Bassanite' , 
                              'Kaolinite' , 'Al Smectite' , 'Margarite' , 'Illite Muscovite' , 'Fe Smectite' , 'Mg Smectite' , 'Talc' , 'Serpentine' , 'Chlorite' , 
                              'Mg Carbonate' , 'Ca Fe Carbonate' , 
                              'Prehnite' , 'Hydrated Silica' , 'Epidote' , 'Analcime' , 'Chloride']
    
        if mineralname not in MICA_spectra_names:
            print('The name of the mineral must be in the list named MICA_spectra_names and the name must be present also in the spectra_MICA_LAB_info.csv file along with all other required inputs!')
            return
    
        # This is here just because at the moment I do not have any laboratory spectra of these minerals
        if mineralname == 'CO2 Ice' or mineralname == 'Hydroxylated Fe-Sulfate':
            use = 'MICA'

        # Check if the use parameter is one of these
        if use not in ['MICA' , 'LAB' , 'Both']:
                raise ValueError("Parameter must be either 'MICA' , 'LAB' or 'Both'.")

        # Upload the .csv files with all the indicized products
        if self.folder == None:
            spectra_info = pd.read_csv('spectra_MICA_LAB_info.csv' , sep = None ,
                                       names = ["Mineral Name","Mineral Type","txt name CRISM","txt name LAB","CRISM spectra cube","Lab. Mineral",
                                                "Type of Lab. From MICA","MICA Lab.Code","Type of Lab. From website",
                                                "Notable Absorptions CRISM [nm]","Notable Absorptions LAB [nm]",
                                                "Sample Grain Size [mum]"])
        else:
            spectra_info = pd.read_csv(self.folder + '/' + 'spectra_MICA_LAB_info.csv' , sep = None ,
                                       names = ["Mineral Name","Mineral Type","txt name CRISM","txt name LAB","CRISM spectra cube","Lab. Mineral",
                                                "Type of Lab. From MICA","MICA Lab.Code","Type of Lab. From website",
                                                "Notable Absorptions CRISM [nm]","Notable Absorptions LAB [nm]",
                                                "Sample Grain Size [mum]"])
            
        self.index_db = spectra_info[spectra_info['Mineral Name'] == mineralname]
        
        # Constraint to one index
        return self.index_db
        
    def simple_compare(self, mineralname, smooth = False, use = 'LAB', alpha = 0.15, folder = None, errorplot = False):
        """
        Function to do a simple spectra compariSON using both the MICA spectra and the laboratory spectra.
        To do so it simply plot the smoothed or non-smoothed normalized spectrum and it superimposes the absorption lines
        of the mineral guess given in the input.
        IMPORTANT: to use this function one must have the spectra_MICA_LAB_info.csv file and all the files it points to in the same folder!.
        
        Parameters
        ----------
        mineralname : string, case sensitive 
            Name of the mineral as given in the Mineral Name column in the spectra_MICA_LAB_info.csv file.
        smooth : bool
            If True it uses the last smoothed spectrum.
        use : string {'MICA' , 'LAB' , 'Both'} 
            Which reference to use for the comparison. For the first it uses the MICA spectra fro comaprison, for 'LAB' it uses the laboratory spectra and for 'Both' it uses both.
        alpha : float
            Shade strength of the error plot if errorplot is True. Default is 0.15.
        folder : string
            Folder in which the 'spectra_MICA_LAB_info.csv' is located. If None the folder is defaulted as the home fodler. Default is None.
        errorplot : bool 
            If to plot or not the errorbar fo the normalized spectrum. Default is False.
            
        Returns
        -------
        None
        """
        
        I = self.index_db#self.dataset_interaction(mineralname , use)

        # Compute the spectra limits for the plot
        spectra_lim = self.limits()

        # Upload MICA spectra
        if use == 'MICA' or use == 'Both':

            CRISM_spectra = np.genfromtxt(I['txt name CRISM'].iloc[0])
            CRISM_spectra_x = CRISM_spectra[:,0]*1000     
            CRISM_lim = self.limits(other_w = CRISM_spectra_x)
            abs_CRISM = eval(I['Notable Absorptions CRISM [nm]'].iloc[0])

        # Upload laboratory spectra
        if use == 'LAB' or use == 'Both':

            LAB_spectra = np.genfromtxt(I['txt name LAB'].iloc[0])
            LAB_spectra_x = LAB_spectra[:,0]*1000
            LAB_spectra_y = LAB_spectra[:,1]

            if I['Type of Lab. From website'].iloc[0] == 'USGS':
                LAB_spectra_x = LAB_spectra[:,0]/10000
                LAB_spectra_y = LAB_spectra[:,1]/1e7

            LAB_lim = self.limits(other_w = LAB_spectra_x)
            abs_LAB = eval(I['Notable Absorptions LAB [nm]'].iloc[0])

        w = self.w[spectra_lim[0] : spectra_lim[1]]
        
        if smooth == True:
            norm  , normerr = self.final_smooth , np.zeros(len(self.final_smooth))
        else:
            norm , normerr = self.final , self.final_error

        if errorplot != False:
            plt.plot(w , norm+normerr , 'k--' , linewidth = 1)
            plt.plot(w , norm-normerr , 'k--' , linewidth = 1)
            plt.fill_between(w , norm-normerr , norm+normerr , color = 'k' , alpha = alpha)

        plt.plot(w , self.final , color = 'k' , linewidth = 2)

        if use == 'Both' or use == 'MICA':
            for i in range(len(abs_CRISM)):
                    if abs_CRISM[i] != None:
                        plt.axvline(abs_CRISM[i] , color = 'r' , linestyle = '--' , linewidth = 1)

        if use == 'Both' or use == 'LAB':
            for i in range(len(abs_LAB)):
                if abs_LAB[i] != None:
                    plt.axvline(abs_LAB[i] , color = 'b' , linestyle = '--' , linewidth = 1)

        plt.xlim(self.MIN , self.MAX)
        plt.xlabel('$\lambda$ [nm]' , fontsize = 15)
        plt.title(mineralname + '. Using ' + use + ' to compare.' , fontsize = 15)
        plt.show()
        
    def compare_spectra(self, mineralname, smooth = False, errorplot = False, use = 'LAB',
                        alpha = 0.15 , folder = None , size = [3,15] ,
                        save = False , namesave = None , folder_save = None , ext = '.png' , show = True):
        """
        Function to do the spectra comparing using both the MICA spectra and the laboratory spectra. \\
        IMPORTANT: to use this function one must have the spectra_MICA_LAB_info.csv file and all the files it points to in the same folder!.
        
        Parameters
        ----------
        mineralname : string
            Name of the mineral as given in the Mineral Name column in the 'spectra_MICA_LAB_info.csv' file.
        smooth : bool
            If True it uses the last smoothed spectrum.
        errorplot : bool 
            If to plot or not the errorbar fo the normalized spectrum. Default is False.
        use : string {'MICA' , 'LAB' , 'Both'} 
            Which reference to use for the comparison. For the first it uses the MICA spectra fro comaprison, for 'LAB' it uses the laboratory spectra and for 'Both' it uses both.
        alpha : float
            Shade strength of the error plot if errorplot is True. Default is 0.15.
        folder : string
            Folder in which the 'spectra_MICA_LAB_info.csv' is located. If None the folder is defaulted as the home fodler. Default is None.
        size : list with two float values
            Size of the plot.
        save : bool 
            To save or not the plot.
        namesave : string
            The name with thich you want to save the plot, if save is True a name must be given.
        folder_save : string 
            The folder path in which you want to save the plot, if None it will save the plot in the home folder.
        ext : string {'.png' , '.jpg' , '.pdf'}
            Extension of the image.
        show : bool
            To show or not the plot.

        Returns
        -------
        None
        """
        
        if smooth == True:
            norm  , normerr = self.final_smooth , np.zeros(len(self.final_smooth))
        else:
            norm , normerr = self.final , self.final_error

        I = self.index_db

        # Compute the spectra limits for the plot
        spectra_lim = self.limits()

        # Upload MICA spectra
        if use == 'MICA' or use == 'Both':
            if folder == None:
                CRISM_spectra = np.genfromtxt(I['txt name CRISM'].iloc[0])
            else:
                CRISM_spectra = np.genfromtxt(folder + '/' + I['txt name CRISM'].iloc[0])
            CRISM_spectra_x = CRISM_spectra[:,0]*1000
            CRISM_spectra_y = CRISM_spectra[:,1]
            CRISM_lim = self.limits(other_w = CRISM_spectra_x)
            abs_CRISM = eval(I['Notable Absorptions CRISM [nm]'].iloc[0])

        # Upload laboratory spectra
        if use == 'LAB' or use == 'Both':
            if folder == None:
                LAB_spectra = np.genfromtxt(I['txt name LAB'].iloc[0])
            else:
                LAB_spectra = np.genfromtxt(folder + '/' + I['txt name LAB'].iloc[0])
            LAB_spectra_x = LAB_spectra[:,0]*1000
            LAB_spectra_y = LAB_spectra[:,1]

            if I['Type of Lab. From website'].iloc[0] == 'USGS':
                LAB_spectra_x = LAB_spectra[:,0]/10000
                LAB_spectra_y = LAB_spectra[:,1]/1e7

            LAB_lim = self.limits(other_w = LAB_spectra_x)
            abs_LAB = eval(I['Notable Absorptions LAB [nm]'].iloc[0])

        # Initialize figure
        fig = plt.figure(figsize = size)

        # Add subplots
        if use == 'Both':
            gs = fig.add_gridspec( 3 , hspace = 0 )
        else:
            gs = fig.add_gridspec( 2 , hspace = 0 )

        w = self.w[spectra_lim[0] : spectra_lim[1]]

        # Make plots share the x axis and set title
        axs = gs.subplots( sharex = True )
        fig.suptitle(mineralname)

        # Setting given x axis limits
        for ax in axs:
            ax.set_xlim(self.MIN , self.MAX)

        # Plot spectra
        axs[0].plot( w , norm , 'k-' , label = 'Target Spectra' )

        # If chosen, plot the error as a shade with dotted borders
        if errorplot == True:
            axs[0].plot( w , norm+normerr , 'k--' , linewidth = 0.5 )
            axs[0].plot( w , norm-normerr , 'k--' , linewidth = 0.5 )
            
            axs[0].fill_between(w , norm-normerr , norm+normerr , color = 'k' , alpha = alpha)
            
            axs[0].set_ylim( np.min(norm-normerr)-0.01 , np.max(norm+normerr) + 0.01 )
            
        else:
            
            axs[0].set_ylim( np.min(norm)-0.01 , np.max(norm) + 0.01 )

        # Plot the MICA and laboratory spectra given the 'use' parameter set to 'Both'
        if use == 'Both':
            axs[1].plot( CRISM_spectra_x , CRISM_spectra_y , 'r-' , label = 'MICA spectra' )  
            axs[2].plot( LAB_spectra_x , LAB_spectra_y , 'b-' , label = I['Type of Lab. From website'].iloc[0] + ' spectra' )

            # Set y axis limits for the MICA and laboratory plot and the labels
            axs[1].set_ylim( np.min(CRISM_spectra_y[CRISM_lim[0] : CRISM_lim[1]])-0.001 , np.max(CRISM_spectra_y[CRISM_lim[0] : CRISM_lim[1]]) + 0.001 )
            axs[2].set_ylim( np.min(LAB_spectra_y[LAB_lim[0] : LAB_lim[1]])-0.001 , np.max(LAB_spectra_y[LAB_lim[0] : LAB_lim[1]]) + 0.001 )
            axs[1].set_ylabel('Relative Reflectance')
            axs[2].set_ylabel('Reflectance')

            # Plot the vertical lines in correspondance with the uploaded absorptions
            for i in range(len(abs_CRISM)):
                for ax in axs:
                    if abs_CRISM[i] != None:
                        ax.axvline(abs_CRISM[i] , color = 'r' , linestyle = '--' , linewidth = 0.5)
                    if abs_LAB[i] != None:
                        ax.axvline(abs_LAB[i] , color = 'b' , linestyle = '--' , linewidth = 0.5)

        # Plot just the MICA spectra given the 'use' parameter set to 'MICA'
        elif use == 'MICA':
            axs[1].plot( CRISM_spectra_x , CRISM_spectra_y , 'r-' , label = 'MICA spectra' )
            axs[1].set_ylabel('Relative Reflectance')

            # Plot the vertical lines in correspondance with the uploaded absorptions
            for i in range(len(abs_CRISM)):
                for ax in axs:
                    if abs_CRISM[i] != None:
                        ax.axvline(abs_CRISM[i] , color = 'r' , linestyle = '--' , linewidth = 0.5)

            axs[1].set_ylim( np.min(CRISM_spectra_y[CRISM_lim[0] : CRISM_lim[1]]) - 0.001 , np.max(CRISM_spectra_y[CRISM_lim[0] : CRISM_lim[1]]) + 0.001 )

        # Plot just the laboratory spectra given the 'use' parameter set to 'LAB'
        else:
            axs[1].plot( LAB_spectra_x , LAB_spectra_y , 'b-' , label = I['Type of Lab. From website'].iloc[0] + ' spectra' )
            axs[1].set_ylabel('Relative Reflectance')
            for i in range(len(abs_LAB)):
                for ax in axs:
                    if abs_LAB[i] != None:
                        ax.axvline(abs_LAB[i] , color = 'b' , linestyle = '--' , linewidth = 0.5)

            axs[1].set_ylim( np.min(LAB_spectra_y[LAB_lim[0] : LAB_lim[1]]) - 0.001 ,
                             np.max(LAB_spectra_y[LAB_lim[0] : LAB_lim[1]]) + 0.001 )

        # Hide x labels and tick labels for all but bottom plot
        for ax in axs:
            ax.label_outer()
            ax.legend()

        # Set label of x axis
        axs[-1].set_xlabel('$\lambda$ [nm]' , fontsize = 15)

        # If chosen, save the figure in given folder with given name
        if save == True and namesave != None:

            if folder_save != None:
                plt.savefig(folder_save + '/' + namesave + ext)
            else:
                plt.savefig(savename + ext)

        if show == True:
            plt.show()
            
    def compare_lines(self, name, smooth = False, ran = 10 , s = 5 , t = 10):    
        """
        Function that automatically infers the nearest absorption lines that are present within a set range
        from the tabulated absorption line position of the given mineral guess from the MICA siles CRISM reference spectra.
        It works by firstly search for local minima in the set interval, if it does not find any sufficiently good
        local minima it searches for inflection points using the KneeLocator function of the kneed module (https://pypi.org/project/kneed/).
        
        Parameters
        ----------
        name : string
            Name of the mineral as given in the Mineral Name column in the 'spectra_MICA_LAB_info.csv' file.
        smooth : bool
            If True it uses the last smoothed spectrum.
        ran : float
            Interval semi-width for the check of elbow points. Deafult is 10.
        s : float
            Sensibility of the elbow finding function. Default is 5.
        t : float
            Threshold limit value for proximity with tabulated absorption. Default is 10.
            
        Returns
        -------
        absoC : list
            Position of the inferred lines from the comparison with MICA CRISM reference spectra.
        absoL : list
            Position of the inferred lines from the comparison with the Laboratory reference spectra.
        diffC : list
            Difference between the inferred lines and the MICA CRISM reference absorption lines.
        diffL : list
            Difference between the inferred lines and the Laboratory reference absorption lines.
        CRISM_abs : list
            MICA CRISM reference absorption lines.
        LAB_abs : list
            LAboratory reference absorption lines.
        """
        
        if smooth == True:
            norm  , normerr = self.final_smooth , np.zeros(len(self.final_smooth))
        else:
            norm , normerr = self.final , self.final_error
        
        spectra_lim = self.limits()

        W = self.w

        W2 = W[spectra_lim[0]:spectra_lim[1]]

        I = self.index_db

        LAB_spectra = np.genfromtxt(I['txt name LAB'].iloc[0])
        CRISM_spectra = np.genfromtxt(I['txt name CRISM'].iloc[0])

        if I['Type of Lab. From website'].iloc[0] == 'USGS':
            LAB_spectra_x = LAB_spectra[:,0]/1000
            LAB_spectra_y = LAB_spectra[:,1]/1e7
        else:
            LAB_spectra_x = LAB_spectra[:,0]*1000
            LAB_spectra_y = LAB_spectra[:,1]

        CRISM_spectra_x = CRISM_spectra[spectra_lim[0]:spectra_lim[1],0]*1000
        CRISM_spectra_y = CRISM_spectra[spectra_lim[0]:spectra_lim[1],1]

        CRISM_abs = abs_CRISM = eval(I['Notable Absorptions CRISM [nm]'].iloc[0])#eval(spectra_info[spectra_info['Mineral Name']==name]['Notable Absorptions CRISM [nm]'].iloc[0])
        LAB_abs = abs_CRISM = eval(I['Notable Absorptions LAB [nm]'].iloc[0])#eval(spectra_info[spectra_info['Mineral Name']==name]['Notable Absorptions LAB [nm]'].iloc[0])

        ab = argrelextrema(norm, np.less)

        absoC , absoL , diffC , diffL = [] , [] , [] , []

        G = 0
        for i in range( len(CRISM_abs) ):
            C = CRISM_abs[i]
            L = LAB_abs[i]
            if C != None and C >= self.MIN and C <= self.MAX:
                A , Ai = np.zeros( len(ab[0]) ) , np.zeros( len(ab[0]) )
                for j in range(len(ab[0])):
                    ww , h = W2[ab[0][j]] , norm[ab[0][j]]
                    A[j] = np.abs( ww - C )
                    Ai[j] = ab[0][j]

                k , ki = np.min(A) , np.argmin(A)

                F = int(Ai[ki])

                Gold = G
                G = W2[F]

                if np.abs(G-Gold) > t:
                    absoC.append(W2[F])
                    absoL.append(W2[F])
                    diffC.append(np.abs(W2[F]-C))
                    diffL.append(np.abs(W2[F]-L))
                else:
                    E = self.find_nearest(W2 , C)
                    Elbow = KneeLocator(W2[E-ran:E+ran], norm[E-ran:E+ran], S=s, curve="convex", direction="decreasing")
                    absoC.append(Elbow.elbow)
                    absoL.append(Elbow.elbow)
                    diffC.append(np.abs(Elbow.elbow - C))
                    diffL.append(np.abs(Elbow.elbow - L))

            elif C == None and L != None and L >= self.MIN and L <= self.MAX:

                A , Ai = np.zeros( len(ab[0]) ) , np.zeros( len(ab[0]) )
                for j in range(len(ab[0])):
                    ww , h = W2[ab[0][j]] , self.final[ab[0][j]]
                    A[j] = np.abs( ww - L )
                    Ai[j] = ab[0][j]

                k , ki = np.min(A) , np.argmin(A)

                F = int(Ai[ki])

                Gold = G
                G = W2[F]

                if np.abs(G-Gold) > t:
                    absoL.append(W2[F])
                    diffL.append(np.abs(W2[F]-L))
                else:
                    E = self.find_nearest(W2 , L)
                    Elbow = KneeLocator(W2[E-ran:E+ran], hyd1[E-ran:E+ran], S=s, curve="convex", direction="increasing")
                    absoL.append(Elbow.knee)
                    diffL.append(np.abs(Elbow.knee-L))
        
        self.absoC = absoC
        self.absoL = absoL
        self.diffC = diffC
        self.diffL = diffL
        
        return self.absoC , self.absoL , self.diffC , self.diffL , CRISM_abs , LAB_abs
    
    def compare2(self , p , n_spectra , n_err , spectra , mineralname , smooth = False , save = False , fold = None , name = None):
        """
        Function to show the comparison using the inferred absorption
        lines found using SpectraAnalysis.compare_lines(). This comparison will show three plots,
        one showing the target and neutral spectrum with the respective errors,
        one showing the (non)smooth ratioed spectrum and the MICA files CRISM
        spectrum with the inferred absorption lines and the last one being the
        laboratory spectrum with the inferred absorption lines.
        
        Parameters
        ---------
        p : float
            Separator for the double plot of the target and MICA spectra.
        mineralname : string
            Name of the mineral as given in the Mineral Name column in the 'spectra_MICA_LAB_info.csv' file.
        smooth : bool
            If True it uses the last smoothed spectrum.
        save : string
            If to save the plot or not. Default is False.
        fold : string
            If save is True this is the path in which to save the plot. If None then the home directory is selected. Default is None.
        name : string
            Name of the plot if required to save it. Default is None.
            
        Returns
        -------
        None
        """

        if smooth == True:
            norm  , normerr = self.final_smooth , np.zeros(len(self.final_smooth))
        else:
            norm , normerr = self.final , self.final_error

        spectra_lim = self.limits()
        W2 = self.w[spectra_lim[0]:spectra_lim[1]]

        spectra_info = pd.read_csv('spectra_MICA_LAB_info.csv' , sep = None ,
                                   names = ["Mineral Name","Mineral Type","txt name CRISM","txt name LAB","CRISM spectra cube","Lab. Mineral",
                                            "Type of Lab. From MICA","MICA Lab.Code","Type of Lab. From website",
                                            "Notable Absorptions CRISM [nm]","Notable Absorptions LAB [nm]",
                                            "Sample Grain Size [mum]"])

        I = self.index_db

        LAB_spectra = np.genfromtxt(I['txt name LAB'].iloc[0])

        if I['Type of Lab. From website'].iloc[0] == 'USGS':
            LAB_spectra_x = LAB_spectra[:,0]/1e4
            LAB_spectra_y = LAB_spectra[:,1]/1e7
        else:
            LAB_spectra_x = LAB_spectra[:,0]*1000
            LAB_spectra_y = LAB_spectra[:,1]

        CRISM_spectra = np.genfromtxt(I['txt name CRISM'].iloc[0])
        CRISM_spectra_x = CRISM_spectra[:,0]*1000
        CRISM_spectra_y = CRISM_spectra[:,1]

        mean_spec = self.m_spec[spectra_lim[0]:spectra_lim[1]]
        std_spec = self.err_spec[spectra_lim[0]:spectra_lim[1]]
        n_spectra = self.n_spectra[spectra_lim[0]:spectra_lim[1]]
        n_err = self.n_err[spectra_lim[0]:spectra_lim[1]]

        #############################################################################################################
        #############################################################################################################
        #############################################################################################################

        fig , ax = plt.subplots(1 , 3 , figsize = [15,5])

        ax[0].plot(W2 , mean_spec , color = 'blue' , linestyle = '-' , linewidth = 1 , label = 'Mean Target Spectrum')
        ax[0].plot(W2 , mean_spec+std_spec , color = 'blue' , linestyle = '--' , linewidth = 0.5)
        ax[0].plot(W2 , mean_spec-std_spec , color = 'blue' , linestyle = '--' , linewidth = 0.5)
        ax[0].fill_between(W2 , mean_spec-std_spec , mean_spec+std_spec , color = 'blue' , alpha = 0.2)
        ax[0].plot(W2 , n_spectra , 'k-' , linewidth = 1 , label = 'Median Neutral Spectrum')
        ax[0].plot(W2 , n_spectra+n_err , 'k--' , linewidth = 0.5)
        ax[0].plot(W2 , n_spectra-n_err , 'k--' , linewidth = 0.5)
        ax[0].fill_between(W2 , n_spectra-n_err , n_spectra+n_err , color = 'k' , alpha = 0.2)
        ax[0].set_xlabel( '$\lambda$ [nm]' , fontsize = 10 )
        ax[0].set_ylabel( 'Reflectance' , fontsize = 10 )
        ax[0].set_xlim(self.MIN , self.MAX)

        if np.min(n_spectra-n_err) <= np.min(mean_spec-std_spec):
            m = np.min(n_spectra-n_err)
        else:
            m = np.min(mean_spec-std_spec)
        if np.max(n_spectra+n_err) <= np.max(mean_spec+std_spec):
            M = np.max(mean_spec+std_spec)
        else:
            M = np.max(n_spectra+n_err)

        ax[0].set_ylim(m,M)

        ax[0].xaxis.set_tick_params(labelsize=10 , rotation = 90)
        ax[0].set_xticks(np.arange(self.MIN , self.MAX+200 , 200))
        ax[0].xaxis.set_minor_locator(MultipleLocator(100))

        ax[0].legend()

        #############################################################################################################
        #############################################################################################################
        #############################################################################################################

        CRISM_spectra_x = CRISM_spectra_x[spectra_lim[0]:spectra_lim[1]]
        CRISM_spectra_y = CRISM_spectra_y[spectra_lim[0]:spectra_lim[1]]

        CRISM_abs = eval(spectra_info[spectra_info['Mineral Name']==mineralname]['Notable Absorptions CRISM [nm]'].iloc[0])
        ax1=ax[1].twinx()
        
        ax1.plot(W2 , norm+np.ones(len(norm))*p , color = 'blue' , linestyle = '-' , linewidth = 1 , label = '$\frac{Target}{Neutral}$')
        
        ax[1].plot(CRISM_spectra_x , CRISM_spectra_y , color = 'k' , linestyle = '-' , linewidth = 1 , label = 'MICA ' + mineralname)

        ax[1].set_xlabel( '$\lambda$ [nm]' , fontsize = 10 )
        ax[1].set_ylabel( 'Relative Reflectance' , fontsize = 10 )
        ax1.set_ylabel( 'Relative Reflectance' , fontsize = 10 , color = 'blue' )
        ax1.yaxis.set_tick_params(colors = 'blue')
        ax[1].set_xlim( self.MIN , self.MAX )

        if np.min(norm+np.ones(len(norm))*p) <= np.min(CRISM_spectra_y):
            m = np.min(norm+np.ones(len(norm))*p)
        else:
            m = np.min(CRISM_spectra_y)
        if np.max(norm+np.ones(len(norm))*p) <= np.max(CRISM_spectra_y):
            M = np.max(CRISM_spectra_y)
        else:
            M = np.max(norm+np.ones(len(norm))*p)

        ax[1].set_ylim(m,M)
        ax[1].xaxis.set_tick_params(labelsize=10 , rotation = 90)
        ax[1].set_xticks(np.arange(self.MIN , self.MAX+200 , 200))
        ax[1].xaxis.set_minor_locator(MultipleLocator(100))

        ax[1].legend()

        #############################################################################################################
        #############################################################################################################
        #############################################################################################################

        LAB_abs = eval(spectra_info[spectra_info['Mineral Name']==mineralname]['Notable Absorptions LAB [nm]'].iloc[0])

        #ax2=ax[2].twinx()
        #ax2.plot(W2 , self.final , color = 'b' , linestyle = '-' , linewidth = 1)# , label = 'Target Spectrum')

        ax[2].plot(LAB_spectra_x , LAB_spectra_y , color = 'k' , linestyle = '-' , linewidth = 1)
        ax[2].set_xlabel( '$\lambda$ [nm]' , fontsize = 10 )
        ax[2].set_ylabel( 'Reflectance' , fontsize = 10 )
        ax[2].set_xlim(self.MIN , self.MAX)
        ax[2].xaxis.set_tick_params(labelsize=10 , rotation = 90)
        ax[2].set_xticks(np.arange(self.MIN , self.MAX+200 , 200))
        ax[2].xaxis.set_minor_locator(MultipleLocator(100))

        #############################################################################################################
        #############################################################################################################
        #############################################################################################################

        for i in range(len(self.absoC)):
            ax[1].axvline(self.absoC[i] , color = 'red' , linestyle = '--' , linewidth = 1)

        for i in range(len(self.absoL)):
            ax[2].axvline(self.absoL[i] , color = 'red' , linestyle = '--' , linewidth = 1)

        fig.subplots_adjust(wspace=0.5)

        if save == True:
            plt.savefig(fold + '/' + name + '.png')
        else:
            plt.show()

    def convex_hull(self , interp = 'linear' , plot = False):
        """
        Function to apply the convex hull caseline correction before feeding a spectrum to the mgm.
        
        Parameters
        ----------
        interp : string
        
        plot : bool
            To plot or not the result. Default is False.
        
        Returns
        -------
        final : numpy 1-d array
            baseline-corrected spectrum. WARNING: THIS FUNCTION RE-INITIALIZE THE self.final SPECTRUM!
        """
        I , J = self.find_nearest(self.w , self.MIN) , self.find_nearest(self.w , self.MAX)
        x , y = self.w[I:J] , self.final[I:J]
        points = np.c_[x , y]
        augmented = np.concatenate([points, [(x[0], np.min(y)-1), (x[-1], np.min(y)-1)]], axis=0)
        hull = ConvexHull(augmented , incremental = True)
        continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
        continuum_indexs = np.array(range(0,len(continuum_points) , 1) , dtype = int)

        continuum_function = interp1d(*continuum_points.T , kind = interp)

        hull = continuum_function(x)
        
        self.final = y/hull
        
        if plot == True:
            fig , ax = plt.subplots(1,2)
            ax[0].plot(x , y , 'k')
            ax[0].plot(x , hull , 'b')
            ax[0].set_xlabel('$\lambda$ [nm]')
            ax[1].set_xlabel('$\lambda$ [nm]')
            ax[1].plot(x , self.final , 'k')
            plt.show()

        return self.final        
    
    def mgm(self , n, iterations, lr, means=None, mean_ranges = [] , asym=False ,
            smooth = False , hull = False ,
            interp = 'linear' , t = 'savgol' , w = 0 , o = 0):
        r"""
        Machine learning driven Modified Gaussian Model (MGM, https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/JB095iB05p06955?src=getftr).
        This version of the algorithm implements gradient-descent, thus it efficiently fits the reference spectrum with the required number of skew normal distributions (https://en.wikipedia.org/wiki/Skew_normal_distribution) formulated as follows:
        
        .. math::
        
            G(\lambda;\alpha , \mu , A , f) = A\cdot e^{-4ln(2)\left(\frac{\lambda - \mu}{f}\right)^{2}}\cdot\left[1 + erf\left(2\sqrt{ln(2)}\alpha\frac{\lambda - \mu}{f}\right)\right]
        
        Where λ is the array of wavelengths, α is the skewing parameter, μ is the mean (i.e. the line position), A is the amplitude and f is the full width at half maximum.
        The used loss function is Pytorch.F Mean Square Loss (MSE, https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html).
        
        Parameters
        ----------
        n : int
            Number of skew normal distributions to be fitted.
        max_iterations : int
            Number of iterations.
        lr : float
            Learning rate for the Adam orptimizer.
        means : list of float or None
            If None, then the means (μ) are inserted in the fitting parameters. If a list is given, then the values in the list will be used as mean values for the distributions. Default is None.
        mean_ranges : list of tuples of float or None
            If None, no ranges is given to the distributions for the fit. If the list of tuples is given, then the search of the mean will be constrained inside an interval (min,max).
        asym : bool
            If True, then the distributions fitted are Skew Normal Distributions, if False, then the alpha parameters will be set to zero, thus efficiently modeling the absorptions as simple Gaussian distributions. Default is False.
        smooth : bool
            Whether or not to smooth the spectrum before the run. Default is False.
        hull : bool
            Whether or not to correct by baseline the spectrum before the run. This can be useful when working
            with spectra not obtained thourgh the convex hull normalization method, sine their values can be higher
            than one, exceeding so the maximum value fot this function. Default is False
        interp : string
            Type of interpolation, can be ‘linear’, ‘nearest’, ‘nearest-up’,
            ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. Default is ‘linear’.
        t : string
            Type of smoothing algorithm to be used. Must be either ‘savgol’ (Savitzky-Golay filter) 
            or ‘movmean’ (Moving Average filter). Default is ‘savgol’.
        w : int
            Smoothing window size. Default is 0.
        o : int
            Polynomial order for the Savitzky-Golay smoothing filter, can be between 1 and 5. Default is 0.
            
        Returns
        -------
        losses : array
            Array of the loss value for each epoch.
        gaussian_sums : array
            Resulting fit per epoch.
        n_wav : array
            Normalized wavelength array.
        gaussian_sum : array
            Final fit.
        a : list
            Amplitude (A) parameters of the distributions.
        m : list
            Mean (μ) parameters of the distributions.
        fwhm : list
            Full width at half maximum parameters (f) of the distributions.
        alpha : list
            Skew parameters (α) of the distributions.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lims = self.limits()
        wavelength = self.w[lims[0]:lims[1]]
        self.w = wavelength
        
        if hull == True:
            self.final = self.convex_hull(interp = interp , plot = True)
            self.w = self.w[0:len(self.w)-1]
        
        if smooth == True:
            if t == 'savgol':
                self.final = self.savgol(w , o , limiti = False)
            elif t == 'movmean':
                self.final = self.moving_average(w , limiti = False)
    
        data = self.final

        n_wav = np.zeros_like(self.w)
        for i in range(len(self.w)):
            n_wav[i] = (self.w[i] - min(self.w)) / (max(self.w) - min(self.w))

        data = torch.tensor(self.final, dtype=torch.float32, device=device)
        wavelength = torch.tensor(self.w, dtype=torch.float32, device=device)
        A , B = max(self.w) , min(self.w)
        n_wav = torch.tensor(n_wav, dtype=torch.float32, device=device)

        #d = min(data)
        #L , l = np.log((1-d)/(d)) , np.log(1/( (1/lr)-1))
        #print(L , l)
        #LL = np.random.uniform(low=l, high=L, size=n)
        #a = torch.tensor(LL).requires_grad_()
        
        a = torch.randn(n, device=device, requires_grad=True)

        fwhm = torch.randn(n, device=device, requires_grad=True)

        #if means is None:
        #    m = torch.randn(n, device=device, requires_grad=True)
        #    params = [a, m, fwhm]
        #else:
        #    params = [a, fwhm]
        #    for i in range(len(means)):
        #        means[i] = (means[i] - B) / (A-B)
        #    m = torch.tensor(np.array(means), dtype=torch.float32, device=device)

        if means is None:
            if mean_ranges is not None:
                # Normalize and create tensor bounds
                mean_bounds = [( (min_val - B) / (A - B), (max_val - B) / (A - B) ) for (min_val, max_val) in mean_ranges]
                m_raw = torch.rand(n, device=device, requires_grad=True)  # between 0 and 1
                params = [a, m_raw, fwhm]
            else:
                m_raw = torch.randn(n, device=device, requires_grad=True)
                params = [a, m_raw, fwhm]
        else:
            params = [a, fwhm]
            for i in range(len(means)):
                means[i] = (means[i] - B) / (A-B)
            m = torch.tensor(np.array(means), dtype=torch.float32, device=device)

        if asym:
            alpha = torch.randn(n, device=device, requires_grad=True)
            params.append(alpha)
        else:
            alpha = torch.zeros(n, device=device)

        optimizer = torch.optim.Adam(params, lr=lr)

        losses = []
        gaussian_sums = []

        #print('Initial Conditions are: \n',
        #      'A = ', torch.sigmoid(a).cpu().detach().numpy(), '\n',
        #      'FWHM = ', (torch.sigmoid(fwhm) * (A-B)).cpu().detach().numpy(), '\n',
        #      'Alpha = ', alpha.cpu().detach().numpy(), '\n',
        #      'X = ', (m * (A-B) + B).cpu().detach().numpy(), '\n')

        # Compute m2 once to show in print
        if means is None:
            if mean_ranges is not None:
                m2 = torch.zeros_like(m_raw)
                for idx in range(n):
                    min_bound, max_bound = mean_bounds[idx]
                    m2[idx] = min_bound + (max_bound - min_bound) * torch.sigmoid(m_raw[idx])
            else:
                m2 = torch.sigmoid(m_raw)
        else:
            m2 = m
        
        print('Initial Conditions are: \n',
              'A = ', torch.sigmoid(a).cpu().detach().numpy(), '\n',
              'FWHM = ', (torch.sigmoid(fwhm) * (A-B)).cpu().detach().numpy(), '\n',
              'Alpha = ', alpha.cpu().detach().numpy(), '\n',
              'M = ', (m2 * (A-B) + B).cpu().detach().numpy(), '\n')

        K , k = 2 * np.sqrt(2 * np.log(2)) , 2*np.sqrt(np.log(2))

        for i in range(iterations):
            optimizer.zero_grad()

            a2 = torch.sigmoid(a)
            fwhm2 = torch.sigmoid(fwhm)
            #if means is None:
            #    m2 = torch.sigmoid(m)
            #else:
            #    m2 = m

            if means is None:
                if mean_ranges is not None:
                    m2 = torch.zeros_like(m_raw)
                    for idx in range(n):
                        min_bound, max_bound = mean_bounds[idx]
                        m2[idx] = min_bound + (max_bound - min_bound) * torch.sigmoid(m_raw[idx])
                else:
                    m2 = torch.sigmoid(m_raw)
            else:
                m2 = m

            gaussian_sum = torch.zeros_like(data)
            
            for j in range(n):       
                wavo = (n_wav - m2[j]) * K / fwhm2[j]
                er = torch.erf(alpha[j] * (k * (n_wav - m2[j]) / (fwhm2[j])))
                gaussian_sum += a2[j] * torch.exp( -0.5 * wavo * wavo ) * ( torch.ones_like(er) + er )

            gaussian_sum = torch.ones_like(data) - gaussian_sum

            loss = F.mse_loss(gaussian_sum, data)
            losses.append(loss.cpu().detach().numpy())
            gaussian_sums.append(gaussian_sum.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

        gaussian_sum = gaussian_sum.cpu().detach().numpy()
        a = torch.sigmoid(a).cpu().detach().numpy()
        fwhm = (torch.sigmoid(fwhm)*(A-B)).cpu().detach().numpy()
        #if means is None:
        #    m = (torch.sigmoid(m)*(A-B) + B).cpu().detach().numpy()
        #else:
        #    m = (m*(A-B)+B).cpu().detach().numpy()
        alpha = alpha.cpu().detach().numpy()


        if means is None:
            if mean_ranges is not None:
                m = []
                for idx in range(n):
                    min_bound, max_bound = mean_ranges[idx]
                    min_bound , max_bound = (min_bound-B)/(A-B) , (max_bound-B)/(A-B)
                    #min_bound , max_bound = 1/(1+np.exp(-min_bound)) , 1/(1+np.exp(-max_bound))#torch.sigmoid(min_bound) , torch.sigmoid(max_bound)
                    m_val = min_bound + (max_bound - min_bound) * torch.sigmoid(m_raw[idx])
                    m.append(m_val * (A - B) + B)
                m = np.array([v.cpu().detach().numpy() for v in m])
            else:
                m = (torch.sigmoid(m_raw)*(A-B) + B).cpu().detach().numpy()
        else:
            m = (m*(A-B)+B).cpu().detach().numpy()
        
        self.losses = losses
        self.gaussian_sums = gaussian_sums
        self.n_wav = n_wav
        self.gaussian_sum = gaussian_sum
        self.a = a
        self.m = m
        self.fwhm = fwhm
        self.alpha = alpha

        return self.losses, self.gaussian_sums , self.n_wav , self.gaussian_sum, self.a, self.m, self.fwhm, self.alpha

    def create_animation(self , losses, gaussian_sums, data, n_wav, frame_step=10 , FPS = 15 , save = False , path = None , name = 'mgm_gradient_descent'):
        """
        Function to create animation of the fitting process for the SpectraAnalysis.mgm() function.
        
        Parameters
        ----------
        losses : list or array
            Losses per epoch.
        gaussian_sums : 2-dim array
            MGM fit per epoch.
        data : list or array
            Normlized/smoothed spectrum.
        n_wav : list or array
            Normalized wavelengths.
        frame_step : int
            Frame step for the animation. Default is 20.
        FPS : int 
            Photograms frequency for the animation. Default is 15.
        save : bool
            If to save or not the animation. Default is False.
        path : string
            Path to which to save the animation. If None then the home directory is used. Default is None.
        name : string
            Name of the animation if wanted to be saved. Default is mgm_gradient_descent.
        
        Returns
        -------
        None
        """
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1]} , figsize = [20,10])

        ax[1].set_ylabel('log(loss)')
        ax[1].set_xlabel('Epoch')
        ax[0].set_xlabel('Normalized Wavelength')
        line_gaussian, = ax[0].plot([], [], 'b')
        line_data, = ax[0].plot(self.n_wav, self.final, 'r', linewidth=2)
        line_loss, = ax[1].plot([], [], 'k')

        def init():
            ax[0].set_xlim(0, 1.1)
            ax[0].set_ylim(np.min(self.final), np.max(self.final))
            ax[1].set_xlim(0, len(self.losses))
            ax[1].set_ylim(min(self.losses), max(self.losses))

            ax[1].set_yscale('log')
            return line_gaussian, line_data, line_loss

        def update_plot(frame):
            i = frame * frame_step
            if i >= len(self.losses):
                return

            line_gaussian.set_data(self.n_wav, self.gaussian_sums[i])
            ax[0].set_ylim(np.min(self.gaussian_sums[i]), np.max(self.final)+0.01)#np.max(gaussian_sums[i]))
            line_loss.set_data(np.arange(0, i+1, frame_step), losses[:i+1:frame_step])

            return line_gaussian, line_data, line_loss
    
        num_frames = len(self.losses) // frame_step
        ani = animation.FuncAnimation(fig, update_plot, frames=num_frames, init_func=init, interval=50, blit=True, repeat=False)
        if save == True:
            writervideo = animation.PillowWriter(fps=FPS, metadata=dict(artist='Me'),bitrate=1800)
            ani.save(path + name + '.gif', writer=writervideo)
        plt.show()

    def mse(self):
        r"""
        Calculates the mean square error:
        
        .. math::
        
            mse = (y_{fit} - y)^{2}
        Parameters
        ----------
        None
        
        Returns
        -------
        
        residuals : array
            Mean square error of the fit per wavelength.
        """
        residuals = np.zeros(len(self.gaussian_sum))
        for i in range(len(self.gaussian_sum)):
            residuals[i] = ( self.gaussian_sum[i] - self.final[i] )**2
        self.residuals = residuals
        return self.residuals
    
    def global_fit(self):
        """
        Function to plot the fit of the machine learning MGM with the resulting mean square error per wavelength on top.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        def scientific_format(y, pos):
            return r'${0:.1f} \times 10^{{-6}}$'.format(y * 1e6)
        
        #lims = self.limits()
        wav = self.w#[lims[0]:lims[1]]
        
        fig , ax = plt.subplots(2,1 , sharex = True , squeeze = True ,gridspec_kw={'width_ratios':[1], 'height_ratios':[1,4]})
        mr , Mr = min(self.residuals) , max(self.residuals)
        #ax[0].set_yscale('log')
        ax[1].plot(wav , self.final , 'k' , label = 'Data')
        ax[1].plot(wav , self.gaussian_sum , 'b' , label = 'Approximation')
        ax[0].plot(wav , self.residuals , 'r' , label = 'Residuals')
        
        ax[0].yaxis.set_major_formatter(FuncFormatter(scientific_format))

        ax[1].set_xlabel('$\lambda$ [nm]')

        fig.subplots_adjust(hspace=0)

        ax[1].legend()
        ax[0].set_ylabel('MSE')#log(MSE)')
        plt.show()
    
    def local_fit(self):
        """
        Function to plot the fit of each single distribution of the machine learning MGM.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        #lims = self.limits()
        wav = self.w#[lims[0]:lims[1]]
        
        ones = np.ones(len(wav))
        GAUSS = np.zeros((len(self.a) , len(wav)))
        for i in range(len(self.a)):
            
            W = np.exp( -4*np.log(2)*( (wav - self.m[i]*ones)/self.fwhm[i] )**2 )
            skew = erf( 2*np.sqrt(np.log(2))*self.alpha[i]*( wav - self.m[i]*ones )/self.fwhm[i]  )
            
            GAUSS[i] = self.a[i]*W*(ones+skew)
    
        fig , ax = plt.subplots()
        ax.plot(wav , self.final , 'r' , linewidth = 1.5)
        for i in range(len(self.a)):            
            ax.plot(wav , ones - GAUSS[i] , 'b--' , linewidth = 1)
        ax.set_xlabel('$\lambda$ [nm]')
        plt.show()
        
        return GAUSS

    def both_fit(self):
            """
            Function to plot the fit of each single distribution, the global fit and the MSE per wavelength of the machine learning MGM.
            
            Parameters
            ----------
            None
            
            Returns
            -------
            None
            """
            a = self.a
            m = self.m
            fwhm = self.fwhm
            alpha = self.alpha
            
            #lims = self.limits()
            wav = self.w#[lims[0]:lims[1]]
            
            ones = np.ones(len(wav))
            GAUSS = np.zeros((len(self.a) , len(wav)))
            for i in range(len(self.a)):
                
                W = np.exp( -4*np.log(2)*( (wav - self.m[i]*ones)/self.fwhm[i] )**2 )
                skew = erf( 2*np.sqrt(np.log(2))*self.alpha[i]*( wav - self.m[i]*ones )/self.fwhm[i]  )
                
                GAUSS[i] = self.a[i]*W*(ones+skew)
    
            def scientific_format(y, pos):
                return r'${0:.1f} \times 10^{{-6}}$'.format(y * 1e6)
            
            #lims = self.limits()
            wav = self.w#[lims[0]:lims[1]]
            
            fig , ax = plt.subplots(2,1 , sharex = True , squeeze = True ,gridspec_kw={'width_ratios':[1], 'height_ratios':[1,4]})
            mr , Mr = min(self.residuals) , max(self.residuals)
            #ax[0].set_yscale('log')
            for i in range(len(self.a)):            
                if i == len(self.a)-1:
                    ax[1].plot(wav , ones - GAUSS[i] , 'b--' , linewidth = 1 , label = 'Single Distribution')
                else:
                    ax[1].plot(wav , ones - GAUSS[i] , 'b--' , linewidth = 1)
            ax[1].plot(wav , self.final , 'k' , label = 'Target')
            ax[1].plot(wav , self.gaussian_sum , 'b' , label = 'Approximation')
            ax[0].plot(wav , self.residuals , 'r' , label = 'Residuals')
            
            ax[0].yaxis.set_major_formatter(FuncFormatter(scientific_format))
    
            ax[1].set_xlabel('$\lambda$ [nm]')
    
            fig.subplots_adjust(hspace=0)
    
            ax[1].legend()
            ax[0].set_ylabel('MSE')
            ax[1].set_ylabel('Relative Reflectance')
            plt.show()
            
            return GAUSS
            
            #lims = self.limits()
            wav = self.w#[lims[0]:lims[1]]
            
            ones = np.ones(len(wav))
            GAUSS = np.zeros((len(self.a) , len(wav)))
            for i in range(len(self.a)):
                
                W = np.exp( -4*np.log(2)*( (wav - self.m[i]*ones)/self.fwhm[i] )**2 )
                skew = erf( 2*np.sqrt(np.log(2))*self.alpha[i]*( wav - self.m[i]*ones )/self.fwhm[i]  )
                
                GAUSS[i] = self.a[i]*W*(ones+skew)
        
            fig , ax = plt.subplots()
            ax.plot(wav , self.final , 'r' , linewidth = 1.5)
            for i in range(len(self.a)):            
                ax.plot(wav , ones - GAUSS[i] , 'b--' , linewidth = 1)
            ax.set_xlabel('$\lambda$ [nm]')
            plt.show()
            
            return GAUSS

class MaficAnalysis():
    """
    Adpated from the paper from Horgan et al., 2014 ( https://doi.org/10.1016/j.icarus.2014.02.031 ).
    This class is taylored for the analysis of mafic materials and mixes.

    Parameters
    ---------
    data : numpy array
        Data (i.e. mean spectrum) to be analyzed.
    wav : numpy array
        Wavelengths used as x-coordinate.
    MIN : float
        Minimum wavelength value
    MAX : float
        Maximum wavelength value
    fs : float
        Font size of text of axis and legends in the plots.
    """
    
    def __init__(self, spectrum, wavelength, MIN, MAX, fsize = 15, pre_normed = True):
        self.data = spectrum
        self.wav = wavelength
        self.MIN = MIN
        self.MAX = MAX
        self.fs = fsize
        self.pre_normed = pre_normed
    
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def plot(self):
        plt.plot(self.wav , self.data , 'k')
        plt.axvline(self.MIN , color = 'red' , linestyle = '--')
        plt.axvline(self.MAX , color = 'blue' , linestyle = '--')
        plt.xlabel('$\lambda$[nm]' , fontsize = self.fs)
        plt.ylabel('Reflectance' , fontsize = self.fs)
        plt.show()

    def continuum_removal(self, interptype='linear', plot=False , smooth = False):
        """
        Applies continuum removal using convex hull or flat model for out-of-range spectra.
        The hull will be constructed from the beginning to the maximum point of the spectrum. After that the hull will be a flat line.
    
        Parameters
        ----------
        interptype : string
            Interpolation type for continuum modeling
        plot : bool
            Whether to plot the result. Default is False.
        smooth : bool
            Whether the input is the smoothed spectrum or not. Default is False.
    
        Returns
        -------
        removed_data : array
            Continuum-removed reflectance.
        """
        if self.pre_normed == True:
            self.removed_data = self.data
            #return self.data
        
        I , J = self.find_nearest(self.wav , self.MIN) , self.find_nearest(self.wav , self.MAX)
        
        if smooth == True:
            x , y = self.wav[I:J] , self.data
        else:
            x , y = self.wav[I:J] , self.data[I:J]

        points = np.c_[x, y]
        augmented = np.concatenate([points, [(x[0], np.min(y) - 1), (x[-1], np.min(y) - 1)]], axis=0)
        hull = ConvexHull(augmented, incremental=True)
        continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
        continuum_function = interp1d(*continuum_points.T, kind=interptype)
        continuum = continuum_function(x)
    
        if y[-1] < np.max(y):
            max_idx = np.argmax(y)
            x_max, y_max = x[max_idx], y[max_idx]
        
            # First part: from start to max
            points1 = np.c_[x[:max_idx + 1], y[:max_idx + 1]]
            augmented1 = np.concatenate([
                points1,
                [(x[0], np.min(y) - 1), (x[max_idx], np.min(y) - 1)]
            ])
        
            hull = ConvexHull(augmented1, incremental=True)
            left_points = points1[np.sort([v for v in hull.vertices if v < len(points1)])]
        
            # Second part: flat line from max to end
            x_flat = x[max_idx:]
            y_flat = np.full_like(x_flat, y_max)
        
            # Combine both
            x_cont = np.concatenate((left_points[:, 0], x_flat))
            y_cont = np.concatenate((left_points[:, 1], y_flat))
        
            continuum = interp1d(x_cont, y_cont, kind=interptype)(x)
            
        else:
            points = np.c_[x, y]
            augmented = np.concatenate([points, [(x[0], np.min(y) - 1), (x[-1], np.min(y) - 1)]], axis=0)
            hull = ConvexHull(augmented, incremental=True)
            continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
            continuum_function = interp1d(*continuum_points.T, kind=interptype)
            continuum = continuum_function(x)
    
        self.wav_cut = x
        self.data_cut = y
        self.continuum = continuum
        self.removed_data = y / continuum
    
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(x, y, 'k', label='Original Spectrum')
            ax[0].plot(x, continuum, 'r' , label='Continuum')
            ax[1].plot(x, self.removed_data, 'b', label='Continuum-Removed')
            for a in ax:
                a.legend(fontsize = self.fs)
                a.set_xlabel('Wavelength (nm)' , fontsize = self.fs)
                a.set_ylabel('Reflectance' , fontsize = self.fs)
            plt.tight_layout()
            plt.show()
    
        return self.removed_data

    def bands_removal(self , windows , mins , maxs , plot = False):
        """
        Bands removal technique involving a moving average over a given subset(s) of the spectrum.

        Parameters
        ----------
        windows : list
            List of the smoothing window to apply to the subset(s).
        mins : list
            List of the minimum values at which the smoothing process needs to be applied.
        maxs : list
            List of the maximum values at which the smoothing process needs to be applied.
        plot : bool
            Whether to plot the result. Default is False.

        Returns
        -------
        spectrum : array
            Cleaned spectrum. IMPORTANT, THIS OVERWRITES THE INPUT DATA.
        """

        for i in range(len(windows)):
            win = windows[i]
            I , J = self.find_nearest(self.wav , mins[i]) , self.find_nearest(self.wav , maxs[i])

            for i in range(I,J+1):
                start = max(0, i - win)
                end = min(len(self.wav), i + win + 1)
                self.data[i] = np.mean(self.data[start:end])
    
        if plot == True:
            if self.pre_normed == True:
                I , J = self.find_nearest(self.wav , self.MIN) , self.find_nearest(self.wav , self.MAX)
                plt.plot(self.wav[I:J] , self.data , 'k')
            else:
                plt.plot(self.wav , self.data , 'k')
            plt.ylabel('Reflectance', fontsize = self.fs)
            plt.xlabel('$\lambda$[nm]', fontsize = self.fs)
            plt.show()
        return self.data
    
    def moving_average(self , window_size, plot = False, overwrite = False, removed = True):
        """
        Function to perform a moving average smoothing on the normalized spectrum.
        
        Parameters
        ----------
        window_size : int
            Size of the step taken for the moving mean.
        plot : bool
            Whether to plot the result. Default is False.
        overwrite : bool
            Whether or not to overwrite the previsou self.data. Default is False.
        removed : bool
            Whether or not the data underwent continuum removal process or not. Default is True.
        
        Returns
        -------
        result : array
            Moving-mean smoothed normalized spectrum.
        """
            
        if window_size < 1:
            raise ValueError("Window_size must be at least 1.")

        I , J = self.find_nearest(self.wav , self.MIN) , self.find_nearest(self.wav , self.MAX)
        x = self.wav[I:J]
        if self.pre_normed == False:
            if removed == True:
                y = self.removed_data
            else:
                y = self.data[I:J]
        else:
            y = self.data
        half_window = window_size // 2
        result = np.zeros_like(y)

        for i in range(len(y)):
            start = max(0, i - half_window)
            end = min(len(x), i + half_window + 1)
            result[i] = np.mean(y[start:end])
        self.final_smooth = result

        if plot == True:
            plt.plot(x , y , 'b')
            plt.plot(x , self.final_smooth , 'r')
            plt.xlabel('$\lambda$[nm]', fontsize = self.fs)
            if removed == True or self.pre_normed == True:
                plt.ylabel('Relative Reflectance', fontsize = self.fs)
            else:
                plt.ylabel('Reflectance', fontsize = self.fs)
            plt.show()

        if overwrite == True:
            self.data = self.final_smooth
        return self.final_smooth

    def band_parameters(self, windows_nm = 75 , resolution_nm = 5 , plot = False , tol = 10 , smoothed_after_removed = False):
        """
        Core function of the class. Adapted from Horgan et al., 2014 (https://doi.org/10.1016/j.icarus.2014.02.031).
        The band parameters are the band minimum, band center, band depth, band area and band asymmetry. For more infor refer to the paper.
        The function can sometime not catch up and returns some error. Try with other parameters configuration in case. 
        The issue refers to the way band centers are computed: due to the interpolation, sometimes that interpolation does not have a minium in the
        give  region around the band minimum.

        Parameters
        ----------
        windows_nm : float
            Nanometric window around the band minimum where to search the band center.
            
        resolution_nm : float
            Nanometric resolution of the 4-th order polynomial approximation used for the band center determination.

        plot : bool
            Whether to plot the result. Default is False.

        tol : float
            Minimum band wavelength span in nanometers. Bands with less wavelength span will not be analyzed.

        Returns
        -------
        parameters : list
            List of the parameters for each band.
        """
        I , J = self.find_nearest(self.wav , self.MIN) , self.find_nearest(self.wav , self.MAX)
        x = self.wav[I:J]

        if smoothed_after_removed == True:
            y = self.final_smooth
        else:
            if self.pre_normed == True:
                y = self.data
            else:
                y = self.removed_data

        ones_indexes = np.argwhere(y == 1)#self.removed_data == 1)
        ones_idx = []
        for i in range(len(ones_indexes)):
            ones_idx.append(ones_indexes[i][0])

        print(ones_idx , ones_indexes)
        
        parameters = []

        if plot == True:
            fig , ax = plt.subplots( )
        k = 0
        for i in range(0, len(ones_idx) - 1):
            
            band_parameters = []

            a = ones_idx[i+1]
            if ones_idx[i] == a:
                continue

            if ones_idx[i+1] - ones_idx[i] >= tol:
                k += 1
            
                S = y[ones_idx[i]:ones_idx[i+1]]
                X = x[ones_idx[i]:ones_idx[i+1]]

                band_parameters.append( (min(X) , max(X)) )
    
                # minimum computation
                minimum , minimum_index = np.min(S) , np.argmin(S)
                
                band_parameters.append(X[minimum_index])
    
                # center computation
                shift_right = self.find_nearest(X , X[minimum_index]+windows_nm)
                shift_left = self.find_nearest(X , X[minimum_index]-windows_nm)
                Xfit_range = np.arange(X[shift_left] , X[shift_right]+resolution_nm , resolution_nm)
                
                interp_S = interp1d(X, S, kind='cubic')(Xfit_range)
                
                coeffs = np.polyfit(Xfit_range, interp_S, 4)
                poly = np.poly1d(coeffs)
                y_poly = poly(Xfit_range)
                
                idx_center = np.argmin(y_poly)#[shift_left:shift_right])
                band_center_wav = Xfit_range[idx_center]
                band_center_val = y_poly[idx_center]
                
                band_parameters.append(band_center_wav)
    
                # depth computation
                band_depth = 1 - S[idx_center]
                
                band_parameters.append(band_depth)

                # area and asymmetry computation (open spectrum compatible)
                if X[-1] >= x[-1] - resolution_nm:
                    # Open band: skip area and asymmetry
                    band_parameters.append(None)  # total_area
                    band_parameters.append(None)  # asymmetry
                else:
                    total_area = np.trapz(np.ones_like(S) - S, X)
                    band_parameters.append(total_area)
                
                    left_area = np.trapz(S[:idx_center], X[:idx_center])
                    right_area = np.trapz(S[idx_center:], X[idx_center:])
                    asymmetry = (right_area - left_area) / (100 * total_area)
                    band_parameters.append(asymmetry)

                band_parameters.append(asymmetry)

                if plot == True:
                    ax.axvline(band_center_wav+5 , color = 'blue' , linestyle = '--' , label = 'Band Center' , linewidth = 1)
                    ax.axvline(X[minimum_index]+5 , color = 'red' , linestyle = '--' , label = 'Band Minimum' , linewidth = 1)
                    ax.fill_between(X, np.ones(len(S)) , S , color = 'grey' , alpha = 0.5)
                    ax.plot(Xfit_range , y_poly , color = 'black' , marker = 'o')

                parameters.append(band_parameters)
                print('Band number ' ,  k)
                print('Band extremes = ' , band_parameters[0] , ' nm')
                print('Band minimum = ' , np.round(band_parameters[1] , 2) , ' nm')
                print('Band center = ' , np.round(band_parameters[2] , 2) , ' nm')
                print('Band depth = ' , np.round(band_parameters[3] , 2))
                print('Band area = ' , np.round(band_parameters[4] , 2) , ' nm')
                print('Band asymmetry = ' , np.round(band_parameters[5] , 2) , ' %')
                print('--------------------------------')

        if plot == True:
            ax.plot(x , y , 'k' , label = 'Continuum Removed Smoothed Spectrum')
            ax.set_xlabel('$\lambda$[nm]' , fontsize = self.fs)
            ax.set_ylabel('Relative Reflectance' , fontsize = self.fs)
            plt.tick_params(axis='both', labelsize=self.fs)

            minimum = plt.Line2D([], [], color='red', linestyle='--', linewidth = 1 , label = 'Band Minimum')
            center = plt.Line2D([], [], color='blue', linestyle='--', linewidth = 1 , label = 'Band Center')
            spectrum = plt.Line2D([], [], color='black', linestyle='-', linewidth = 1 , label = 'Target Spectrum')
            interpolation = plt.Line2D([], [], color='black', marker = 'o', linestyle=None , markersize = 10 , label = '4-th Degree Polynomial')
            
            ax.legend(handles=[minimum , center , spectrum , interpolation][::-1] , fontsize = self.fs)
            
            #plt.legend(fontsize = self.fs)
            plt.show()

        return parameters
