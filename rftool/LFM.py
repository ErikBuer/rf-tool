import scipy.signal as signal
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.special as special
import scipy.ndimage as ndimage
import numpy as np
import numpy.polynomial.polynomial as poly

from mpl_toolkits.mplot3d import axes3d     # 3D plot
import matplotlib.pyplot as plt
from matplotlib import cm
colorMap = cm.coolwarm

from pyhht.visualization import plot_imfs   # Hilbert-Huang TF analysis
from pyhht import EMD                       # Hilbert-Huang TF analysis
import rftool.utility as util
import rftool.estimation as estimate

class chirp:
    """
    Object for generating a set of linear chirps with a 
    """
    t = None
    T = None

    def __init__( self, Fs=1e3, T=1, fStart=1, fStop=10, nChirps=16):
        """
        T is the chirp duration [s].
        Fs is the intended sampling frequency [Hz]. Fs must be at last twice the highest frequency in the input PSD. If Fs < 2*max(f), then Fs = 2*max(f)
        nChirps is the number of chirps to be generated
        """
        self.Fs = Fs
        self.T = T
        self.points = np.intc(Fs*T)
        self.dt = 1/self.Fs
        self.fStart = fStart
        self.fStop = fStop
        self.nChirps = np.intc(nChirps)
        self.getPrimaryChirp()

    def getPrimaryChirp(self):
        self.omega_t = np.linspace( self.fStart, self.fStop, np.intc(self.T*self.Fs) )

        # Delay for each symbol in samples
        # Half of the symbols is in one direction, another half in the other (up/down).
        self.symbolDelay = np.linspace(0, self.points-(self.points/(self.nChirps/2))-1, np.intc(self.nChirps/2))
        self.symbolDelay = np.append(self.symbolDelay, self.symbolDelay)

    def getSymbolSig(self, symbol):
        """
        Generate chirps.
        """
        omega_t = np.roll( self.omega_t, np.intc(self.symbolDelay[symbol]) )        

        # The second half of sympols has invertedd chirp direction
        if np.intc(self.nChirps/2)-1<symbol:
            omega_t = np.max(omega_t) - (omega_t-np.min(omega_t))

        phi_t = util.indefIntegration( omega_t, self.dt )
        sig = np.exp(np.multiply(1j*2*np.pi, phi_t))
        return sig

    def getSymbolIF(self, symbol):
        """
        Return the IF of the symbols
        """
        omega_t = np.roll( self.omega_t, np.intc(self.symbolDelay[symbol]) )        

        # The second half of sympols has invertedd chirp direction
        if np.intc(self.nChirps/2)<=symbol:
            omega_t = np.max(omega_t) - (omega_t-np.min(omega_t))
        return omega_t
    
    def plotSymbols(self):
        root = np.intc(np.ceil(np.sqrt(self.nChirps)))
        fig, axs = plt.subplots(root,root)
        for index, axis in enumerate(axs.flat):
            if index<self.nChirps:
                axis.plot(self.getSymbolIF(index), label=str(index))
                axis.legend()
    
    def plotAutocorr(self):
        root = np.intc(np.ceil(np.sqrt(self.nChirps)))
        fig, axs = plt.subplots(root,root)
        plt.title('Autocorrelation')
        for index, axis in enumerate(axs.flat):
            if index<self.nChirps:
                axis.plot( util.pow2normdb(np.abs(signal.correlate( self.getSymbolSig(index), self.getSymbolSig(index) , mode='same', method='fft'))), label=str(index)) 
                axis.legend()
        

    def plotXcorr(self):
        corrmat = np.zeros((self.nChirps,self.nChirps))
        it = np.nditer(corrmat, flags=['multi_index'])
        while not it.finished:
            corrmat[it.multi_index] = np.max( np.abs(signal.correlate( self.getSymbolSig(it.multi_index[0]), self.getSymbolSig(it.multi_index[1]) , mode='same', method='fft')) )
            it.iternext()

        corrmat = util.pow2normdb(corrmat)
        plt.figure()
        plt.title('Cross Corrlation [dB]')
        plt.pcolormesh(corrmat, cmap=colorMap)
        plt.colorbar()
        