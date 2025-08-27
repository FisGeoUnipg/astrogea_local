"""
    SpectraNorm module
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
class SpectraNorm():
    """
    Class to Normalize the mean target spectrum.
    
    Parameters
    ----------
    RGB : 3-dim array or None
        The RGB image to be used. Inside the class it also possible to upload the RGB map, so it is possible to set this parameter as None.
    Nbands : int
        Number of wavelengths used (basically, len(wavelength)).
    img : python spectral.io.bsqfile.BsqFile object
        Spectral reflectances datacube.
    img_sr : python spectral.io.bsqfile.BsqFile object
        Spectral parameters datacube.
    wavelength : list or array
        CRISM observed wavelengths.
    target : array
        The mean target spectrum.
    error_target : array
        The standard deviation of the target spectrum.
    MIN : float
        Minimum wavelength value to be taken into consideration.
    MAX : float
        Maximum wavelength value to be taken into consideration.
    """
    def __init__(self , RGB , Nbands , img , img_sr , wavelength , target , error_target , spectra , MIN , MAX):
        self.RGB = RGB
        self.Nbands = Nbands
        self.img = img
        self.img_sr = img_sr
        self.w = wavelength
        self.target = target
        self.error_target = error_target
        self.spectra = spectra
        self.MIN = MIN
        self.MAX = MAX

    def upload_spectrum(self , name , folder = None , mean = True):
            """
            Function to upload a set of pre-extracted spectra, saved using SpectraExtarct.save_spectra().
            
            Parameters
            ----------
            name : string
                Name of the file that wants to be uploaded, without the format extension.
            folder : string
                Path of the file that want s to be uploaded. If None path is taken as home directory. Default is None.
                
            Returns
            -------
            spectra : 2-dim array
                Uploaded pre-extracted spectra.
            target_spectrum : 1-dim array
                Mean/median of the pre-extracted spectra.
            error_spectrum : 1-dim array
                Standard deviation/median absolute deviation of the pre-extracted spectra.
            """
            if folder == None:
                data = np.genfromtxt(name + '.txt' , dtype = int)
            else:
                data = np.genfromtxt(folder + name + '.txt' , dtype = int)
    
            x , y = data[:,0] , data[:,1]
    
            spectra = np.zeros( ( len(x) , len(self.w) ) )
    
            for i in range(len(x)):
                spectra[i] = np.transpose( self.img[x[i] , y[i] , :] )[:,0,0]
        
            plt.imshow(self.RGB)
            plt.plot(y,x, 'w.')
            plt.show()
    
            if mean == True:
                target_spectrum = np.mean(spectra.T , axis = 1)
                error_spectrum = np.std(spectra.T , axis = 1)
            else:
                target_spectrum = np.median(spectra.T , axis = 1)
                error_spectrum = MAD(spectra.T , axis = 1)
    
            self.m_spec , self.err_spec , self.spectra = target_spectrum , error_spectrum , spectra
    
            return self.spectra , self.m_spec , self.err_spec
        
    def upload_map(self , name , folder = None):
        """
        Function to upload a pre-made RGB map, saved using RGBImageManipulator.save_map(), that wants to be used to extract the spectra.
        
        Parameters
        ----------
        name : string
            Name of the RGB map that wants to be uploaded.
        folder : string
            Path of the RGB map that want s to be uploaded. If None path is taken as home directory. Default is None.
            
        Returns
        -------
        RGB : 3-dim array
            Uploaded RGB map.
        """
        if folder == None:
            R = np.loadtxt(name + '_R.txt')
            G = np.loadtxt(name + '_G.txt')
            B = np.loadtxt(name + '_B.txt')
        else:
            R = np.loadtxt(folder + name + '_R.txt')
            G = np.loadtxt(folder + name + '_G.txt')
            B = np.loadtxt(folder + name + '_B.txt')    
            
        image = np.zeros((R.shape[0] , R.shape[1] , 3))

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j,0] = R[i,j]
                image[i,j,1] = G[i,j]
                image[i,j,2] = B[i,j]
                
        RGB = image

        return RGB
    
    def upload_neutral(self , name , method , folder):
        """
        Function to upload a pre-made median/mad spectrum.
        
        Parameters
        ----------
        name : string
            Name of the file.
        method : string
            It will add a suffix after the name on the base on the method with which the neutral spectra was computed.
        folder : string or None
            If None it will be saved into the home folder, if a folder path is given, the path must end with the /.
        
        Returns
        -------
        med : 1d-array
            The uploaded median spectrum.
        mad : 1d-array
            The MAD of the uplkoaded median spectrum.
        w : 1d-array
            The cut wavelength range.
        """
        
        i , j = self.find_nearest(self.MIN , self.w) , self.find_nearest(self.MAX , self.w)

        if folder == None:
            self.w = np.genfromtxt('Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            self.med = np.genfromtxt(name + '_' + method +'_median.txt')
            self.mad = np.genfromtxt(name + '_' + method +'_mad.txt')
        else:
            self.w = np.genfromtxt(folder + 'Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            self.med = np.genfromtxt(folder + name + '_' + method +'_median.txt')
            self.mad = np.genfromtxt(folder + name + '_' + method +'_mad.txt')
        
        return self.med , self.mad , self.w

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

    def neutral_polygon_spectra(self, c_line='white', c_marker='white', save_pixel=False, folder=None, name='neutral_spectra_coordinates'):
        """
        Function to select the area from which to extract the neutral spectra by drawing a polygon on the RGB image. The spectra are then extracted from the enclosed pixels from the spectral reflectances datacube.
        This function is essentially the same as SpectraExtract.polygon_spectra().
        
        Parameters
        ----------
        c_line : string
            Color of the polygon's sides. Default is 'white'.
        c_marker : string
            Color of the polygon's corners. Default is 'white'.
        save_pixel : bool
            If to save or not the enclosed pixels coordinates in a .txt file. Default is False.
        fodler : string
            Folder in which to save the pixels if save_pixels is True. If None it saves in the home directory. Default is None.
        name : string
            Name with which the pixels coorinates are saved. Default is 'spectra_coordinates'.
            
        Returns
        -------
        spectra : 2-dim array
            Spectra extracted from the spectral reflectance datacube from the pixels enclosed in the polygon drewn on the RGB map.
        med : 1-dim array
            Median of the extracted spectra.
        mad : 1-dim array
            MAD of the extracted spectra.
        L : int
            Amount of pixels selected.
        mask : 2-dim array
            Masked RGB array to enhance polygon position.
        """
        fig, ax = plt.subplots()
        ax.imshow(self.RGB)

        def onselect(poly_verts):
            global poly
            poly = poly_verts

        poly_selector = PolygonSelector(ax , onselect , props = {'color': c_line})
        plt.show()

        path = Path(poly)

        y , x = np.mgrid[:self.RGB.shape[0] , :self.RGB.shape[1]]
        points = np.vstack((x.ravel() , y.ravel())).T
        mask = path.contains_points(points)
        mask = mask.reshape(self.RGB.shape[:2])

        points_inside = self.RGB[mask]
        L = len(points_inside)
        print('Number of points inside the drewn polygon: ' , L)

        indices_1 , indices_2 = np.where(mask)

        spectra = np.zeros( ( L , self.Nbands ) )

        for i in range(len(indices_1)):

            spectra[i] = np.transpose( self.img[indices_1[i] , indices_2[i] , :] )[:,0,0]

        if save_pixel == True:
            if folder == None:
                np.savetxt(name + '.txt' , np.array([indices_2 , indices_1] , dtype = int))
            else:
                np.savetxt(folder + name + '.txt' , np.array([indices_2 , indices_1] , dtype = int))

        self.neutral = spectra

        self.med = np.median(spectra , axis = 0)
        self.mad = MAD(spectra , axis = 0)

        self.neutral_spectra = spectra

        return self.neutral , self.med , self.mad , L , mask
    
    def neutral_convex_hull(self , interp = 'linear'):
        """
        Function to select as neutral spectrum the convex hull (the calculation of is mutuated from scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html).
        
        Parameters
        ---------
        interp : string {‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’}
            Is the interpolation of the convex hull using interp1d from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).
            The possible arguments signify: ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.
        
        Returns
        -------
        neutral : 1-dim array
            Points of the convex hull.
        med : 1-dim array
            Median of the convex hull, in this case is the same as neutral, but for completeness we keeped the same strucutre as the others methods.
        mad : 1-dim array
            MAD of the convex hull, since it only one "spectrum" it is an array filled with zeros.
        """

        I , J = self.find_nearest(self.w , self.MIN) , self.find_nearest(self.w , self.MAX)
        x , y = self.w[I:J] , self.target[I:J]
        points = np.c_[x , y]
        augmented = np.concatenate([points, [(x[0], np.min(y)-1), (x[-1], np.min(y)-1)]], axis=0)
        hull = ConvexHull(augmented , incremental = True)
        continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
        continuum_indexs = np.array(range(0,len(continuum_points) , 1) , dtype = int)

        continuum_function = interp1d(*continuum_points.T , kind = interp)

        self.neutral = continuum_function(x)
        
        self.med = self.neutral
        self.mad = np.zeros(len(self.neutral))

        return self.neutral , self.med , self.mad

    def plot_together(self , convex_hull = False):
        """
        Function to plot together the mean target spectrum +- its standard deviation and the median neutral spectrum +- its MAD.
        
        Parameters
        ----------
        convex_hull : bool
            If the convex hull neutral spectrum is used. This is done since the convex hull spectrum is drewn directly onto the x-cut spectrum and so does not need to be cut in the same range as the target. Default is False.
        
        Returns
        -------
        None
        """

        lims = self.limits()
        a , b = lims[0] , lims[1]

        wav = self.w[a:b]
        
        if convex_hull == False:
            neutral , neutralerr = self.med[a:b] , self.mad[a:b]
        else:
            neutral , neutralerr = self.med , self.mad

        plt.figure()
        plt.plot(wav , neutral , 'k-' , label = 'Median Neutral')
        plt.plot(wav , neutral + neutralerr , 'k--')
        plt.plot(wav , neutral - neutralerr , 'k--')
        plt.fill_between(wav , neutral - neutralerr , neutral + neutralerr , color = 'k' , alpha = 0.2)

        plt.plot(wav , self.target[a:b] , 'b-' , label = 'Mean Target')
        plt.plot(wav , self.target[a:b] + self.error_target[a:b] , 'b--')
        plt.plot(wav , self.target[a:b] - self.error_target[a:b] , 'b--')
        plt.fill_between(wav , self.target[a:b] - self.error_target[a:b] , self.target[a:b] + self.error_target[a:b] , color = 'b' , alpha = 0.2)

        plt.xlabel ('$\lambda$ [nm]')

        plt.show()

    def norm_spectra(self , convex_hull = False):
        r"""
        Normalization of the target spectrum over the neutral. 
        The error propagation formula is used for the resulting final error, 
        and thus can lead to some places having complex error due to the presence 
        of the covariance between the spectra. More advanced methods should 
        be used to evaluate the error in those cases, but for most application it is sufficently good like this.
        Anyway, errors that ends up as complex are set to zero for simplicity.

        Calling :math:`N` the normalized spectrum, :math:`\sigma_N` the resulting error of the normalized spectrum,  
        :math:`A` the target mean spectrum, :math:`B` the median neutral spectrum,  
        :math:`\sigma_A` the standard deviation of the target, :math:`\sigma_B` the MAD of the neutral,  
        and :math:`C_{A,B}` the covariance between the two spectra, the normalization is done in the following way:


        .. math::

            \begin{aligned}
                N &= \frac{A}{B} \\
                \sigma_{N} &= \left| \frac{A}{B} \right| \sqrt{ 
                \left(\frac{\sigma_{A}}{B}\right)^{2} + 
                \left(\frac{\sigma_{B}}{B}\right)^{2} - 
                \frac{2C_{A,B}}{A \cdot B} }
            \end{aligned}
        
        Parameters
        ----------
        convex_hull : bool
            If the normalization is done with the convex hull or not. Default is False.
            
        Returns
        -------
        norm : array
            Normalized spectrum.
        error norm : array
            Propagated error of the normalized error.
        """
        lims = self.limits()
        
        if convex_hull == False:

            A , B , dA , dB = self.target[lims[0]:lims[1]] , self.med[lims[0]:lims[1]] , self.error_target[lims[0]:lims[1]] , self.mad[lims[0]:lims[1]]
        
        else:
            
            A , B , dA , dB = self.target[lims[0]:lims[1]] , self.med , self.error_target[lims[0]:lims[1]] , self.mad
        
        mean , err = np.zeros(len(A)) , np.zeros(len(B))

        C = 0.

        for i in range(len(A)):
            C += ( A[i] - np.mean(A) )*( B[i] - np.mean(B) )

        C = C/len(A)

        k = 0

        for i in range(len(A)):
            mean[i] = A[i]/B[i]
            t0 , t1 , t2 , t3 = np.abs(A[i]/B[i]) , (dA[i]/A[i])**2 , (dB[i]/B[i])**2 , 2*C/(A[i]*B[i])
            if  t1 + t2 - t3 >= 0:
                err[i] = t0*np.sqrt( t1 + t2 - t3 )
            else:
                k += 1

        print(k/len(A)*100 , '% of the errors, set to zero for simplicity, are in reality NaN values.')

        self.norm = mean
        self.normerr = err

        return self.norm , self.normerr
    
    def normplot(self , convex_hull = False):
        """
        Function to plot the normalized spectrum.

        Parameters
        ---------
        None
        
        Returns
        -------
        None
        """
        
        lims = self.limits()
        
        a , b = lims[0] , lims[1]
        
        wav = self.w[a:b]
        
        plt.plot(wav , self.norm , 'k-')
        
        if convex_hull == False:
            plt.plot(wav , self.norm+self.normerr , 'k--')
            plt.plot(wav , self.norm-self.normerr , 'k--')
            plt.fill_between(wav , self.norm-self.normerr , self.norm+self.normerr , color = 'black' , alpha = 0.5)
        
        plt.xlabel('$\lambda$[nm]')
        plt.show()
        
    def save_spectrum(self , name , folder , method , normalized = True):
        """
        Function to save the mean/median and std/MAD spectra and the cut wavelength range.
        
        Parameters
        ----------
        name : string
            Name of the file.
        folder : string or None
            If None it will be saved into the home folder, if a folder path is given, the path must end with the /.
        method : string
            It will add a suffix after the name on the base on the method with which the neutral spectra was computed.
            If method is not ply , sam , mam , min or cxh, this function will not work.
            ply stands for polygon, sam for single allmap, mam for mutiple allmap, min for mineral mask and csh for convex hull.
        normalized : bool
            If True it will also save the normalize spectrum, if False not.

        Returns
        -------
        None
        """
        if method != 'ply' or method != 'sam' or method != 'mam' or method != 'min' or method != 'cxh':
            raise ValueError('method parameter must be one between ply, sam, mam , min or cxh!')
        
        i , j = self.find_nearest(self.MIN , self.w) , self.find_nearest(self.MAX , self.w)
        if folder == None:
            np.savetxt('Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            np.savetxt(name + '_' + method +'_median.txt' , self.m_spec)
            np.savetxt(name + '_' + method +'_mad.txt' , self.err_spec)
            np.savetxt(name + '_' + method +'_norm.txt' , self.err_spec)
        else:
            np.savetxt(folder + 'Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            np.savetxt(folder + name + '_' + method +'_median.txt' , self.m_spec)
            np.savetxt(folder + name + '_' + method +'_mad.txt' , self.err_spec)
            np.savetxt(folder + name + '_' + method +'_norm.txt' , self.err_spec)

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

        data = np.asarray(self.norm)
        result = np.empty(len(self.norm))
        half_window = window_size // 2

        for i in range(len(self.norm)):
            start = max(0, i - half_window)
            end = min(len(self.norm), i + half_window + 1)
            result[i] = np.mean(data[start:end])
        self.final_smooth = result

        if limiti == True:
            lims = self.limits()
            a , b = lims[0] , lims[1]
            plt.plot(self.w[a:b] , self.norm , 'b')
            plt.plot(self.w[a:b] , self.final_smooth , 'r')
        else:
            plt.plot(self.w , self.norm , 'b')
            plt.plot(self.w , self.final_smooth , 'r')
        plt.xlabel('$\lambda$[nm]')
        plt.show()
        
        return self.final_smooth
