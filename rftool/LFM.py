import scipy.signal as signal
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.special as special
import scipy.ndimage as ndimage
import numpy as np
import numpy.polynomial.polynomial as poly

from mpl_toolkits.mplot3d import axes3d     # 3D plot
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

from pyhht.visualization import plot_imfs   # Hilbert-Huang TF analysis
from pyhht import EMD                       # Hilbert-Huang TF analysis
import rftool.utility as util
import rftool.estimation as estimate

class chirp:
    """
    Object for generating a set of linear chirps
    """
    t = None
    T = None

    def __init__( self, Fs=1e3, T=1, fStart=1, fStop=10, nChirps=16, **kwargs):
        """
        T is the chirp duration [s].
        Fs is the intended sampling frequency [Hz]. Fs must be at last twice the highest frequency in the input PSD. If Fs < 2*max(f), then Fs = 2*max(f)
        nChirps is the number of chirps to be generated
        direction is the chirp direction, 'up', 'down', or 'both'
        """
        self.direction = kwargs.get('direction', 'both')
        
        # If 'both' chirp directions are selected, nChirps must be even
        if (0<nChirps%2) & (self.direction=='both'):
            nChirps+=1

        self.Fs = Fs
        self.T = T
        self.points = np.intc(Fs*T)
        self.dt = 1/self.Fs
        self.t = np.linspace(0,T,self.points)
        self.fStart = fStart
        self.fStop = fStop
        self.nChirps = np.intc(nChirps)
        self.getPrimaryChirp()

    def getPrimaryChirp(self):
        self.omega_t = np.linspace( self.fStart, self.fStop, np.intc(self.T*self.Fs) )

        # Delay for each symbol in samples
        # Half of the symbols is in one direction, another half in the other (up/down).
        if self.direction=='both':
            self.symbolDelay = np.linspace(0, self.points-(self.points/(self.nChirps/2)), np.intc(self.nChirps/2))
            self.symbolDelay = np.append(self.symbolDelay, self.symbolDelay)
        else:
            self.symbolDelay = np.linspace(0, self.points-(self.points/(self.nChirps)), np.intc(self.nChirps))

    def getSymbolSig(self, symbol):
        """
        Generate chirps.
        """
        omega_t = self.omega_t

        if self.direction=='down':
            omega_t = np.max(omega_t) - (omega_t-np.min(omega_t))
        elif self.direction=='both':
            # The second half of symbols has invertedd chirp direction
            if np.intc(self.nChirps/2)-1<symbol:
                omega_t = np.max(omega_t) - (omega_t-np.min(omega_t))

        phi_t = util.indefIntegration( omega_t, self.dt )
        sig = np.exp(np.multiply(1j*2*np.pi, phi_t))
        sig = np.roll( sig, np.intc(self.symbolDelay[symbol]) )
        #! Debug code
        """plt.figure()
        plt.plot(sig)
        plt.show()"""
        #! Debug code
        return sig

    def getSymbolIF(self, symbol):
        """
        Return the IF of the symbols
        """
        omega_t = np.roll( self.omega_t, np.intc(self.symbolDelay[symbol]) )

        if self.direction=='down':
            omega_t = np.max(omega_t) - (omega_t-np.min(omega_t))
        elif self.direction=='both':
            # The second half of sympols has invertedd chirp direction
            if np.intc(self.nChirps/2)-1<symbol:
                omega_t = np.max(omega_t) - (omega_t-np.min(omega_t))
        return omega_t
    
    def plotSymbols(self):
        root = np.intc(np.ceil(np.sqrt(self.nChirps)))
        fig, ax = plt.subplots(root,root)
        fig.set_size_inches((7,2.5))
        for index, axis in enumerate(ax.flat):
            if index<self.nChirps:
                axis.plot(self.t, self.getSymbolIF(index), label=str(index))
                #axis.legend()
                axis.set_ylabel('f [Hz]')
                axis.set_xlabel('t [s]')

        plt.tight_layout()
        #fig.suptitle('Chirp Instantaneous Frequency')

    def plotAutocorr(self):
        root = np.intc(np.ceil(np.sqrt(self.nChirps)))
        fig, axs = plt.subplots(root,root)
        fig.figsize=[7, 4]
        #fig.subtitle('Autocorrelation')
        for index, axis in enumerate(axs.flat):
            if index<self.nChirps:
                axis.plot( util.pow2normdb(np.abs(signal.correlate( self.getSymbolSig(index), self.getSymbolSig(index) , mode='same', method='fft'))), label=str(index)) 
                #axis.legend()
        #fig.suptitle('Autocorrelation')
        plt.tight_layout()

    def plotXcorr(self):
        corrmat = np.zeros((self.nChirps,self.nChirps))
        it = np.nditer(corrmat, flags=['multi_index'])
        while not it.finished:
            corrmat[it.multi_index] = np.max( np.abs(signal.correlate( self.getSymbolSig(it.multi_index[0]), self.getSymbolSig(it.multi_index[1]) , mode='same', method='fft')) )
            it.iternext()

        corrmat = util.pow2normdb(corrmat)
        plt.figure(figsize=(3.5, 2.5))
        plt.title('Normalized Cross Corrlation [dB]')
        plt.pcolormesh(corrmat)
        plt.colorbar()
        

    def plotDotProd(self):
        corrmat = np.zeros((self.nChirps,self.nChirps))
        it = np.nditer(corrmat, flags=['multi_index'])
        while not it.finished:
            corrmat[it.multi_index] = np.abs(np.dot(self.getSymbolSig(it.multi_index[0]), self.getSymbolSig(it.multi_index[1])) )
            it.iternext()

        corrmatDb = util.pow2normdb(corrmat)
        plt.figure(figsize=(2.8, 2.1))
        plt.title('Normalized Dot Product [dB]')
        plt.pcolormesh(corrmatDb)
        plt.colorbar()
        return corrmatDb

    def modulate( self, symbolStream=np.array([1,0,1,0])):
        """
        Modulate bit stream to a chirp. One chirp per symbol.
        
        symbolStream is the bitstream to be modulated (numpy array).
        """

        # Calculate length of signal
        sigLen = len(symbolStream)*len(self.omega_t)
        # generate frame
        packetSig = np.empty([sigLen], dtype=complex)
        

        # Iterate through symbolStream and add to packetSig
        for m, symbol in enumerate(symbolStream):
            packetSig[m*self.points:(m+1)*self.points] = self.getSymbolSig(symbol)
        return packetSig