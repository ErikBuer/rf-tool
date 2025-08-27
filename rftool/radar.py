import scipy.signal as signal
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.special as special
import scipy.ndimage as ndimage
import numpy as np
import numpy.polynomial.polynomial as poly

from mpl_toolkits.mplot3d import axes3d     # 3D plot
import matplotlib.pyplot as plt

from pyhht.visualization import plot_imfs   # Hilbert-Huang TF analysis
from pyhht import EMD                       # Hilbert-Huang TF analysis
import rftool.utility as util
import rftool.estimation as estimate

def Albersheim( Pfa, Pd, N ):
    """Calculate required SNR for non-coherent integration over N pulses, by use of Albersheims equation.

    :param Pfa: The probability of false alarm (linear)
    :type Pfa: scalar
    :param Pd: The probability of detection (linear)
    :type Pd: scalar
    :param N: The number of non-coherently integrated pulses
    :type N: Integer
    :return: Required SNR in dB
    :rtype: scalar

    Accurate within 0.2 dB for:
    10^-7   <  Pfa  < 10^-3
    0.1     <  Pd   < 0.9
    1       <= N    < 8096

    - M. A. Richards and J. A. Scheer and W. A. Holm, Principles of Modern Radar, SciTech Publishing, 2010 
    """

    A = np.log(np.divide(0.62, Pfa))
    B = np.log(np.divide(Pd,1-Pd))
    SNRdB = -5*np.log10(N)+(6.2+np.divide(4.54, np.sqrt(N+0.44)))*np.log10(A+(0.12*A*B)+(0.7*B))
    return SNRdB

def Shnidman( Pfa, Pd, N, SW ):
    """Calculate required SNR for non-coherent integration over N pulses for swerling cases, by use of Shnidman equation.

    :param Pfa: The probability of false alarm (linear)
    :type Pfa: scalar
    :param Pd: The probability of detection (linear)
    :type Pd: scalar
    :param N: The number of non-coherently integrated pulses
    :type N: Integer
    :param SW: The Swerling model 0-4 (zero is non-swerling)
    :type SW: Integer
    :return: Required SNR in dB.
    :rtype: scalar

    Accurate within 1 dB for:
    10^-7   <  Pfa  < 10^-3
    0.1     <= Pd   =< 0.99*10^-9
    1       <= N    < 100
    
    - M. A. Richards and J. A. Scheer and W. A. Holm, Principles of Modern Radar, SciTech Publishing, 2010 
    """
    C = 0
    alpha = 0

    if N < 40:
        alpha = 0
    elif N > 40:
        alpha = np.divide(1,4)
    
    eta = np.sqrt( -0.8*np.log(4*Pfa*(1-Pfa)) ) + np.sign(Pd-0.5)*np.sqrt( -0.8*np.log(4*Pd*(1-Pd)) )

    X_inf = eta*( eta + 2*np.sqrt( np.divide(N,2)+(alpha-np.divide(1,4)) ) )

    if SW==0:   # Non swerling case
        C = 1
    else:       # Swerling case
        def K(SW):
            if  SW==1:
                K = 1
            elif  SW==2:
                K = N
            elif  SW==3:
                2
            else:
                K = 2*N

        C1 = np.divide( ( (17.7006*Pd-18.4496)*Pd+14.5339 )*Pd-3.525, K(SW) )
        C2 = np.divide(1,K(SW))*( np.exp( 27.31*Pd-25.14 ) + (Pd-0.8)*( 0.7*np.log( np.divide(1e-5, Pfa) ) + np.divide( N-10, 40 ) ) )
        CdB = 0
        if Pd < 0.8872:
            CdB = C1
        elif Pd > 0.99:
            CdB = C1 + C2
        
        C = np.power(10, np.divide(CdB,10) )
    
    X1= np.divide(C*X_inf, N)
    
    SNRdB = 10*np.log10(X1)
    return SNRdB

def upconvert( sig, f_c, Fs=1 ):
    """Upconvert baseband waveform to IF.
    :param f_c: The IF center frequency.
    :type f_c: scalar
    :param Fs: The sample frequency.
    :type Fs: integer, scalar
    :return: The upconverted signal.
    :rtype: ndarray
    """
    a = np.linspace(0, sig.shape[0]-1, sig.shape[0])
    # Angular increments per sample times sample number
    phi_j = 2*np.pi*np.divide(f_c,Fs)*a
    # Complex IF carrier
    sig = np.multiply(sig, np.exp(1j*phi_j))
    return sig

def ACF(x, Fs, plot = True, **kwargs):
    """Normalized autocorrelation Function of input x.

    :param x: The signal being analyzed. If x is a matrix, the correlation is performed columnwise
    :type x: Vector or matrix, ndarray
    :param Fs: The sample frequency.
    :type Fs: integer, scalar
    :param plot: Decides whether to plot the result, defaults to True
    :type plot: bool, optional
    :param \**kwargs:
        See below
    :return: The output is 2*len(x)-1 long.
    :rtype: Vector or matrix. If the input is a matrix, the output is a matrix (ndarray)


    :Keyword Arguments:
        * *label* (``list``) --
          The plot label for each vector as such: ['label0', 'label1']
    """
    plotLabel = kwargs.get('label', None)

    # If x is a vector, ensure it is a column vector.
    if x.ndim < 2:
        x = np.expand_dims(x, axis=1)

    # Iterate through columns
    r_xx = np.empty( shape=[2*np.size(x,0)-1, np.size(x,1)], dtype=complex )
    for n in range(0, np.size(x,1)):
        #r_xx[:,n] = np.correlate(x[:,n], x[:,n], mode='full')
        r_xx[:,n] = signal.correlate(x[:,n], x[:,n], method='fft')

    if plot == True:
        tau = np.linspace(-np.floor(len(r_xx)/2)/Fs, np.floor(len(r_xx)/2)/Fs, len(r_xx))
        # Plot
        fig, ax = plt.subplots()
        for i, column in enumerate(r_xx.T):
            # Normalize
            column = np.absolute(column / abs(column).max())
            ax.plot(tau, util.mag2db(column), label=plotLabel[i])

        ax.ticklabel_format(useMathText=True, scilimits=(0,3))
        plt.legend()
        yMin = np.maximum(-100, np.min(r_xx))
        ax.set_ylim([yMin, 0])
        #plt.title("Autocorrelation Function")
        ax.set_ylabel("Normalized Correlation [dB]")
        ax.set_xlabel("$t$ [s]")
        plt.tight_layout()
    return r_xx



class chirp:
    """Object for generating linear and non-linear chirp generation.
    """
    t = None
    T = None

    def __init__( self, Fs=1 ):
        """
        :param t_i: The chirp duration [s].
        :type t_i: scalar
        :param Fs: The intended sampling frequency [Hz]. Fs must be at last twice the highest frequency in the input PSD. If Fs < 2*max(f), then Fs = 2*max(f).
        :type Fs: scalar
        """
        self.Fs = Fs
        self.dt = 1/self.Fs

    def checkSampleRate(self):
        """
        Check that the sample rate is within the nyquist criterion. I.e. that no phase difference between two consecutive samples exceeds pi.
        """
        errorVecotr = np.abs(np.gradient(self.targetOmega_t)*2*np.pi)-np.pi

        if 0 < np.max(errorVecotr):
            print("Warning, sample rate too low. Maximum phase change is", np.pi+np.max(errorVecotr), "Maximum allowed is pi." )
    
    def genFromPoly( self, direction = None ):
        """
        Generate Non-Linear Frequency Modualted (NLFM) chirps based on a polynomial of arbitrary order.
        :param direction: Controls the chirp direction. ``'inverted'`` inverts the chirp direction, defaults to None
        :type direction: str
        :return: Chirp
        :rtype: ndarray, time series
        """
        dt = 1/self.Fs        # seconds
        polyOmega = np.poly1d(self.c)

        omega_t = polyOmega(self.t)
        if direction == 'inverted':
            omega_t = np.max(omega_t) - (omega_t-np.min(omega_t))

        phi_t = util.indefIntegration( omega_t, dt )
        sig = np.exp(np.multiply(1j*2*np.pi, phi_t))
        return sig

    def genNumerical( self, direction = None  ):
        """Generate Non.Linear Frequency Modualted (NLFM) chirps.

        :param direction: Controls the chirp direction. ``'inverted'`` inverts the chirp direction, defaults to None
        :type direction: str
        :return: Chirp
        :rtype: ndarray, time series
        """
        dt = 1/self.Fs        # seconds

        omega_t = self.targetOmega_t
        if direction == 'inverted':
            omega_t = np.max(omega_t) - (omega_t-np.min(omega_t))

        phi_t = util.indefIntegration( omega_t, dt )
        sig = np.exp(np.multiply(1j*2*np.pi, phi_t))
        return sig

    def getInstFreq(self, poly=True, plot=True):
        """Calculate the instantaneous frequency as a function of time

        :param poly: Decides whther to use polynomial coefficients, defaults to True
        :type poly: bool, optional
        :param plot: Decides whether to plot the IF, defaults to True
        :type plot: bool, optional
        :return: Instantaneous frequency
        :rtype: ndarray, time series
        """
        if poly == True:
            # Calculate the instantaneous frequency based on polynoimial coefficients.
            polyOmega = np.poly1d(self.c)
            omega_t = polyOmega(self.t)
        else:
            # Calculate the instantaneous frequency based on phase vector.
            omega_t = np.gradient(self.phi_t, self.t)

        if plot == True:
            plt.figure()
            plt.plot(self.t, omega_t)
            plt.plot(self.t, self.targetOmega_t)
            plt.xlabel('t [s]')
            plt.ylabel('f [Hz]')
            plt.title("Instantaneous Frequency")
            plt.show()
        return omega_t

    def getChirpRate(self, poly=True, plot=True):
        """Calculate the chirp rate as a function of time.

        :param poly: Decides if the chirprate is calculated based on polynoimial coefficients, defaults to True
        :type poly: bool, optional
        :param plot: Decides whether to plot the chirp rate, defaults to True
        :type plot: bool, optional
        :return: Chirp rate function
        :rtype: ndarray, time series
        """
        if poly == True:
            # Calculate the chirp rate based on polynoimial coefficients.
            gamma_t = 0
            for n in range(2, len(self.c)):
                polyOmega = np.poly1d(self.c)
                polyGamma = np.polyder(polyOmega)
                omega_t = polyOmega(self.t)
        else:
            # Calculate the chirp rate based on phase vector.
            gamma_t = np.gradient(self.getInstFreq(plot=False), self.t)

        if plot == True:
            plt.figure()
            plt.plot(self.t, gamma_t)
            plt.xlabel('t [s]')
            plt.ylabel('f [Hz]')
            plt.title("Chirp Rate")
            plt.show()
        return gamma_t

    def PSD( self, sig_t, plot=False ):
        """Calculates Power Spectral Density in dBW/Hz.

        :param sig_t: Signal time series.
        :type sig_t: ndarray
        :param plot: Decides whether to plot the PSD, defaults to False
        :type plot: bool, optional
        :return: PSD as a vector (dBW/Hz)
        :rtype: ndarray, vector
        """
        f, psd = signal.welch(sig_t, fs=self.Fs, nfft=self.fftLen, nperseg=self.fftLen, window = signal.blackmanharris(self.fftLen),
        noverlap = self.fftLen/4, return_onesided=False)

        if plot == True:
            #f = np.linspace(-self.Fs/2, self.Fs/2, len(psd))
            # Remove infinitesimally small components
            #psd_dB = util.pow2db(np.maximum(psd, 1e-14))
            psd_dB = util.pow2db(psd)
            plt.plot(f, psd_dB)
            plt.title("Welch's PSD Estimate")
            plt.ylabel("dBW/Hz")
            plt.xlabel("Frequency [Hz]")
            plt.show()
        return psd

    def W( self, omega ):
        """Lookup table for the W function. Takes instantaneous frequency as input.

        :param omega: Insatnaneous frequency vector/time series
        :type omega: ndarray
        :return: Chirp bandwidth
        :rtype: scalar
        """
        delta_omega_W = self.omega_W[1]-self.omega_W[0]

        W_omega = np.empty((len(omega)))
        for i in range(0, len(omega)):
            index = np.intc(omega[i]/delta_omega_W)+np.intc(len(self.window)/2)
            if index<0:
                index = 0
            elif len(self.window)-1<index:
                index = len(self.window)-1

            W_omega[i] = self.window[index]
        return W_omega

    def gamma_t_objective( self, scale ):
        """Objective function for finding gamma_t that meets the constraints.
        scale scales the gamma function.

        :return: Cost
        :rtype: scalar
        """
        self.iterationCount = self.iterationCount +1
        if self.iterationCount % 10 == 0:
            print("Iteration",self.iterationCount)
        
        # Calculate gamma_t for this iteration
        self.gamma_t = self.gamma_t_initial*scale

        # omega_t = integrate.cumtrapz(self.gamma_t, x =self.t ) # resulthas one less cell
        omega_t = np.cumsum(self.gamma_t)*self.dt # Ghetto integral
        # Place center frequency at omega_0
        omega_t = omega_t + (self.omega_0 - omega_t[np.intc(len(omega_t)/2)])
        
        # Scale W function to enclose omega_t
        self.omega_W = np.linspace(omega_t[0]-self.omega_0, omega_t[-1]-self.omega_0, len(self.window))

        # Calculate NLFM gamma function
        self.gamma_t = self.gamma_t[np.intc(len(self.gamma_t)/2)]/self.W(omega_t-self.omega_0)
        self.targetOmega_t = util.indefIntegration(self.gamma_t, self.dt)
        self.targetOmega_t = self.targetOmega_t  + (self.omega_0 - self.targetOmega_t[np.intc(len(self.targetOmega_t)/2)])

        OmegaIteration = np.trapz(self.gamma_t, dx=self.dt)
        cost = np.abs(self.Omega - OmegaIteration)
        return cost

    def getCoefficients( self, window, T=1e-3, targetBw=10e3, centerFreq=20e3, order=48):
        """Calculate the necessary coefficients in order to generate a NLFM chirp with a specific magnitude envelope (in frequency domain). Chirp generated using rftool.radar.generate().
        Coefficients are found through non-linear optimization.

        :param window: The window function for the target PSD. It is used as a LUT based function from -Omega/2 to Omega/2, where Omega=targetBw.
        :type window: ndarray
        :param T: The pulse duration [s]., defaults to 1e-3
        :type T: scalar, optional
        :param targetBw: The taget bandwidth of the chirp [Hz], defaults to 10e3
        :type targetBw: scalar, optional
        :param centerFreq: The center frequency of the chirp [Hz], defaults to 20e3
        :type centerFreq: scalar, optional
        :param order: he order of the phase polynomial used to generate the chirp frequency characteristics [integer], defaults to 48
        :type order: int, scalar, optional
        :return: Polynomial coefficients
        :rtype: ndarray, vector
        """
        self.Omega = targetBw
        self.window = np.maximum(window, 1e-8)
        
        self.omega_0 = centerFreq
        self.T = T
        self.points = np.intc(self.Fs*T)
        self.t = np.linspace(-self.T/2, self.T/2, self.points)
        self.dt = self.T/(self.points-1)

        # optimization routine
        # Count iterations
        self.iterationCount = 0

        # Calculate LFM chirp rate (initial gamma_t)
        self.gamma_t_initial = np.full(self.points, self.Omega/self.T)

        # Initial scaling
        p0=np.array([1])
        # Optimize gamma_t curve with window
        print("Initiating chirp instantaneous frequency optimization.")
        chirpOpt = optimize.minimize(self.gamma_t_objective, p0, method='L-BFGS-B')
        
        # optimization routine
        # Count iterations
        self.iterationCount = 0
        
        # TODO: Remove if functioning properly
        """# Order and initial conditions
        c0 = np.zeros( order )"""
        
        # Resample time series to improve the fitting result.
        omegaFit = signal.decimate(self.targetOmega_t, 16, ftype='iir', zero_phase=True)
        timeFit = np.linspace(-self.T/2, self.T/2, len(omegaFit))

        self.c = np.polyfit(timeFit, omegaFit, order)
        return self.c


    def modulate( self, bitstream=np.array([1,0,1,0])):
        """Modulate bit stream to a chirp. One chirp per bit. A 1 is represented as a forward time chirp.
        A zero is represented as a time-reversed chirp.
        
        bitStream is the bitstream to be modulated (numpy array).

        :param bitstream: [description], defaults to np.array([1,0,1,0])
        :type bitstream: vector, optional
        :return: Modulated waveform
        :rtype: ndarray, time series
        """

        # Generate t and T if it doesn't exist
        if self.T is None:
            self.T = (self.points-1)*self.dt
        if self.t is None:
            self.t = np.linspace(-self.T/2, self.T/2, self.points)

        # Calculate length of signal
        sigLen = len(bitstream)*self.points
        # generate frame
        waveform = np.empty([sigLen], dtype=complex)

        sig = self.genNumerical()
        sigInv = self.genNumerical('inverted')

        # Iterate through bitstream and add to waveform
        for m, bit in enumerate(bitstream):
            if bit==1:
                waveform[m*self.points:(m+1)*self.points] = sig
            elif bit==0:
                waveform[m*self.points:(m+1)*self.points] = sigInv
        return waveform
