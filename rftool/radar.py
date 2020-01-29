import scipy.signal as signal
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import numpy as np
from pyhht.visualization import plot_imfs   # Hilbert-Huang TF analysis
from pyhht import EMD                       # Hilbert-Huang TF analysis
import rftool.utility as util
import timeit       # t&d

def Albersheim( Pfa, Pd, N ):
    """
    Calculate required SNR for non-coherent integration over N pulses, by use of Albersheims equation.

    Pd is the probability of detection (linear)
    Pfa is the probability of false alarm (linear)
    N is the number of non-coherently integrated pulses
    Returns SNR in dB

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
    """
    Calculate required SNR for non-coherent integration over N pulses for swerling cases, by use of Shnidman equation.

    Pd is the probability of detection (linear)
    Pfa is the probability of false alarm (linear)
    N is the number of non-coherently integrated pulses
    SW is the swerling model: 0-4 (zero is non-swerling)
    Returns SNR in dB

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
        C2 = np.divide(1,K(SW))*( np.exp( 27.31*Pd-25.14 ) + (Pd-0.8)*( 0.7*np.log( np.divide(10e-5, Pfa) ) + np.divide( 2*N-20, 80 ) ) )
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
    """
    Upconvert baseband waveform to IF
    f_c is the IF center frequency
    Fs is the sample frequency
    """
    a = np.linspace(0, sig.shape[0]-1, sig.shape[0])
    # Angular increments per sample times sample number
    phi_j = 2*np.pi*np.divide(f_c,Fs)*a
    # Complex IF carrier
    sig = np.multiply(sig, np.exp(1j*phi_j))
    return sig

def hilbert_spectrum( sig, Fs=1 ):
    """
    Hilbert-Huang transform with Hilbert spectral plot.
    The plot neglects negative frequencies.
    
    sig is a time series
    Fs is the sample frequency

    Based on:
    - S.E. Hamdi et al., Hilbert-Huang Transform versus Fourier based analysis for diffused ultrasonic waves structural health monitoring in polymer based composite materials,Proceedings of the Acoustics 2012 Nantes Conference.
    """

    # Hilbert-Huang
    decomposer = EMD(sig)
    imfs = decomposer.decompose()
    #plot_imfs(sig, imfs, t)

    imfAngle = np.angle(signal.hilbert(imfs))
    dt = np.divide(1,Fs)
    
    t = np.linspace(0, (sig.shape[0]-1)*dt, sig.shape[0])

    # Calculate instantaneous frequency
    instFreq = np.divide(np.gradient(imfAngle,t,axis=1), 2*np.pi)
    """
    There is an image of the instantaneous frequency response occuring at -Fs/2. THis is currently not shown in the plot. 
    """

    # Calculate Hilbert spectrum
    # Time, frequency, magnitude

    intensity = np.absolute(signal.hilbert(imfs))
    plt.figure()
    for i in range(np.size(instFreq,0)):
        plt.scatter(t, instFreq[i], c=intensity[i], s=5, alpha=0.3)

    plt.title("Hilbert Spectrum")
    plt.xlabel('t [s]')
    plt.ylabel('f [Hz]')
    plt.ylim(0,np.divide(Fs,2))
    plt.tight_layout()
    plt.show()

def ACF(x, singleSided = True, plot = True):
    """
    Normalized autocorrelation Function of input x
    x is the signal being analyzed.
    singleSided decides wether to return the -inf to inf ACF or 0 to inf ACF.
    plot decides wether to plot the result.
    """
    r_xx = np.correlate(x, x, mode='full')

    if singleSided == True:
        r_xx = r_xx[np.intc(len(r_xx)/2):]

    if plot == True:
        # Normalize
        r_xx = np.absolute(r_xx / abs(r_xx).max())
        # Plot
        plt.figure()
        plt.plot(util.mag2db(r_xx))
        plt.title("Autocorrelation Function")
        plt.ylabel("Normalized magnitude [dB]")
        plt.show()
    return r_xx

# TODO def AF(x, Fs=1, doppler = 2)
    """
    Ambuigity Function
    x is the signal being analyzed.
    Fs is the sampling frequency of the signal x [Hz].
    doppler is the max doppler shift in Hz
    """

class chirp:
    """
    Object for generating linear and non-linear chirps.
    """
    c = np.array([1,1,1])
    # FFT lenth for computation og magnitude spectrum.
    fftLen = 2048

    class target:
        """
        Object for storing the properties of the target chirp. Used in optimization aproach for coefficients.
        """

        def __init__( self,  r_xx_dB=1, f=1, order=6):
            self.r_xx_dB = r_xx_dB
            self.f = f
            self.order = order

    def __init__( self,  t_i=1, Fs=1 ):
        """
        t_i is the chirp duration [s].
        Fs is the intended sampling frequency [Hz]. Fs must be at last twice the highest frequency in the input PSD. If Fs < 2*max(f), then Fs = 2*max(f)
        """
        self.t_i = t_i
        self.Fs = Fs

    def generate( self ):
        """
        Generate Non.Linear Frequency Modualted (NLFM) chirps based on a polynomial of arbitrary order.

        t is the length of the chirp [s]
        c is a vector of phase polynomial coefficients (arbitrary length)
        Fs is the sampling frequency [Hz]

        c[0] is the reference phase
        c[1] is the reference frequency,
        c[2] is the nominal constant chirp rate
        
        For symmetricsl PSD; c_n = 0 for odd n > 2, that is, for n = 3, 5, 7,…

        - A .W. Doerry, Generating Nonlinear FM Chirp Waveforms for Radar, Sandia National Laboratories, 2006
        """
        c = self.c
        dt = np.divide(1,self.Fs)        # seconds
        t = np.linspace(-self.t_i/2, self.t_i/2-dt, np.intc(self.Fs*self.t_i))  # Time vector

        phi_t = np.full(np.intc(self.t_i*self.Fs), c[-1])

        c = np.flip(c)
        c = np.delete(c, 1) # delete c_N
        for c_n in np.nditer(c):
            phi_t = c_n + util.indefIntegration(phi_t, dt)

        gamma_t = np.gradient(phi_t,t)  # Instantaneous frequency
        """
        plt.figure
        plt.plot(t, gamma_t)
        plt.xlabel('t [s]')
        plt.ylabel('f [Hz]')
        plt.title("Instantaneous Frequency")
        plt.show()
        """
        self.sig = np.exp(np.multiply(1j, phi_t))
        return self.sig

    def plotMagnitude( self, sig_t ):
        sig_f = np.fft.fft(sig_t, self.fftLen)/self.fftLen
        # Shift and normalize
        sig_f = np.abs(np.fft.fftshift(sig_f / abs(sig_f).max()))
        # Remove infinitesimally small components
        sig_f = util.mag2db(np.maximum(sig_f, 1e-10))
        f = np.linspace(-self.Fs/2, self.Fs/2, len(sig_f))
        plt.figure()
        plt.plot(f, sig_f)
        plt.xlim([-self.Fs/2, self.Fs/2])
        plt.title("Frequency response")
        plt.ylabel("Normalized magnitude [dB]")
        plt.xlabel("Frequency [Hz]")
        plt.show()

    def PSD( self, sig_t, plot=False ):
        """
        Calculates Power Spectral Density in dBW/Hz.
        """
        f, psd = signal.welch(sig_t, fs=self.Fs, nfft=self.fftLen, nperseg=self.fftLen, return_onesided=False)

        
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

    def ACFdB( self, sig_t, plot=False ):
        """
        Calculates normalized ACF in dB based on FFT downsampling.
        sig_t is the time-domain input signal.
        """
        psd = self.PSD(sig_t)
        r_xx = np.fft.ifft(psd, self.fftLen)
        # Shift and normalize
        r_xx = np.abs(np.fft.fftshift(r_xx / abs(r_xx).max()))
        r_xx_dB = util.mag2db( r_xx )

        if plot == True:
            # Remove infinitesimally small components
            plt.plot(r_xx_dB)
            plt.title("ACF")
            plt.ylabel("Normalized Magnitude [dB]")
            plt.xlabel("Delay")
            plt.show()
        return r_xx_dB

    def getCoefficientsObjectiveFunction( self, c ):
        """
        Objective function for coefficient optimization
        """

        # For symmetricsl PSD; cn = 0 for odd n > 2, that is, for n = 3, 5, 7, …
        # Nulling out these coefficients
        if (self.target.symm == True) and (len(c) > 3):
            
            for i in range(3, len(c)):
                if (i % 2) == 1:
                    c[i]=0

        self.c = c
        sig = self.generate()
        
        # The error vector is the difference in autocorrelation
        errorVector = np.abs(np.subtract(self.target.r_xx_dB, self.ACFdB(sig)))

        cost = np.sum( errorVector )
        return cost

    def getCoefficients( self, r_xx_dB, symm, order=6):
        """
        Calculate the necessary coefficients in order to generate a NLFM chirp with a specific magnitude envelope (in frequency domain). Chirp generated using rftool.radar.generate().
        Coefficients are found through non-linear optimization.

        r_xx_dB is the target aurocorrelation (vector).
        order is the oder of the polynomial used to generate the chirp frequency characteristics. the length of c is order+1.
        symm configures wether the intend PSD should be symmetrical. With a symmetrical target PSD, the result will be close to symmetrical even with symm set to false, 
        however the dimentionality of the problem is reduced greatly by setting symm to True.
        fftLen is the length of the FFT which will be used in comparison between the generated chirp spectral mask and the target mask.
        """
        self.fftLen = len(r_xx_dB)

        # Initiate target chirp object 
        target = self.target(r_xx_dB, order)
        self.target.r_xx_dB = r_xx_dB
        self.target.symm = symm
        
        # optimization routine
        # Inital random search
        best_c = (np.random.rand( order+1 )-0.5)*2e3   # Initial values (random)
        lowestCost = 1e9
        for i in range(1, np.intc(10e3)):
            c = (np.random.rand( order+1 )-0.5)*self.Fs   # Initial values (random)
            cost = self.getCoefficientsObjectiveFunction(c)
            if cost<lowestCost:
                lowestCost = cost
                print("Iteration ", i, "lowestCost = ", lowestCost)
                best_c = c

        c0 = best_c
        

        res = optimize.minimize(self.getCoefficientsObjectiveFunction, c0, tol=1e-6, options={'maxiter': 10000, 'disp': True}) #  method='nelder-mead'
        # Optimized parameters
        # self.c=res.x

        # development stuff
        plt.plot( self.ACFdB(self.sig) )
        plt.plot( self.target.r_xx_dB )
        plt.title("Resulting ACF")
        plt.show()
        return self.c