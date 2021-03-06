import scipy.signal as signal
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.special as special
import scipy.ndimage as ndimage
import scipy.misc as sciMisc
import numpy as np
import numpy.polynomial.polynomial as poly
import tftb as tftb

from mpl_toolkits.mplot3d import axes3d     # 3D plot
import matplotlib.pyplot as plt

from pyhht.visualization import plot_imfs   # Hilbert-Huang TF analysis
from pyhht import EMD                       # Hilbert-Huang TF analysis
import rftool.utility as util

def pulseCarrierCRLB(p_n, Fs, K, l_k, N):
    """
    Calculates the Cramer-Rao Lower Bound for estimation of carrier frequency of a pulse train of unknown coherent pulses.
    Returns CRLB in angular frequency.

    p_n is a single pulse (time series) of appropriate power.
	K is the number of pulses in the pulse train.
	l_k is the discrete pulse times [sample].
    N is a scalar or vector representing the variance of the noise. (Noise power).

    - POURHOMAYOUN, et al., Cramer-Rao Lower Bound for Frequency Estimation for Coherent Pulse Train With Unknown Pulse, IEEE 2013
    """
    # Calculate pulse energy
    E0 = util.energy(p_n)
    # Pulse length
    M = len(p_n)
    # Pulse bandwidth
    B0 = util.energy(np.gradient(p_n, 1/Fs)) / E0
    # Time-frequency cross-coupling (skew)
    C0 = np.imag( np.sum( np.multiply( p_n, np.conj(np.gradient(p_n, 1/Fs)) ) ) )
    R1 = np.mean(l_k[1:])
    R2 = np.mean(np.power(l_k[1:], 2))
    # Variance bound of frequency estimate
    var_theta = np.divide( N, 2*K*(E0-(np.power(C0, 2) / (B0*E0)))*(R2-np.power(R1, 2)) )
    return var_theta

class HilberHuang:
    def __init__( self, sig, Fs=1):
        """
        Hilbert-Huang transform with Hilbert spectral plot.
        
        sig is a time series
        Fs is the sample frequency

        - Huang et. al, The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis, Proceedings of the Royal Society of London, 1998
        - S.E. Hamdi et al., Hilbert-Huang Transform versus Fourier based analysis for diffused ultrasonic waves structural health monitoring in polymer based composite materials, Proceedings of the Acoustics 2012 Nantes Conference.
        """
        self.Fs = Fs

        # Hilbert-Huang
        decomposer = EMD(sig)
        self.imfs = decomposer.decompose()

        analyticalIMF = signal.hilbert(self.imfs)
        imfAngle = np.unwrap(np.angle(analyticalIMF), axis=1)
        dt = np.divide(1,Fs)
        # Calculate Hilbert spectrum
        # Time, frequency, intensity
        self.t = np.linspace(0, (sig.shape[0]-1)*dt, sig.shape[0])
        self.IF = np.divide(np.gradient(imfAngle, self.t,axis=1), 2*np.pi)
        self.intensity = np.absolute(analyticalIMF)

    def discreteMatrix( self, frequencyBins=256, *args, **kwargs):
        """Hilbert-Huang transform with Hilbert spectral plot. The result is a matric of discrete time-frequency bins with the cumulated intensity of the IMFs.
        The plot neglects negative frequencies.

        :param frequencyBins: The number of discrete bins from 0 Hz to Fs/2, defaults to 256
        :type frequencyBins: int, optional
        :param \**kwargs:
            See below
        :return: A matrix containing the energy sum in each time-frequnecy bin
        :rtype: ndarray, matrix

        :Keyword Arguments:
        * *includeRes* (``bool``) -- Defines whether the residual is included in the spectrum, defaults to False

        - Huang et. al, The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis, Proceedings of the Royal Society of London, 1998
        - S.E. Hamdi et al., Hilbert-Huang Transform versus Fourier based analysis for diffused ultrasonic waves structural health monitoring in polymer based composite materials, Proceedings of the Acoustics 2012 Nantes Conference.
        """
        includeRes = kwargs.get('includeRes', False)

        binSize = (self.Fs/2)/frequencyBins
        f = np.linspace(0, frequencyBins*binSize, frequencyBins)

        ImfBin = np.intc(np.floor(np.divide(self.IF, binSize)))
        spectrumMat = np.zeros(( frequencyBins, len(self.t)))

        for m, row in enumerate(ImfBin):
            for n, fBin in enumerate(row):
                if fBin<0:
                    fBin = -fBin
                if frequencyBins-1<fBin:
                    fBin = frequencyBins-1
                spectrumMat[fBin,n] += self.intensity[m,n]

        return f, self.t, spectrumMat

    def discreteSpectrum( self, frequencyBins=256, *args, **kwargs):
        """Hilbert-Huang transform with Hilbert spectral plot. The result is a matric of discrete time-frequency bins with the cumulated intensity of the IMFs.
        The plot neglects negative frequencies.

        :param frequencyBins: Tthe number of discrete bins from 0 Hz to Fs/2, defaults to 256
        :type frequencyBins: int, scalar, optional
        :param \**kwargs:
            See below
        :return: a figure with the sum energy in each time-frequnecy bin.
        :rtype: fig, ax

        :Keyword Arguments:
        * *decimateTime* (``int``) -- Decimates along the time axis to reduce the image size. Input Decimation factor, optional
        * *includeRes* (``bool``) -- Defines whether the residu is included in the spectrum, defaults to False
        
        - Huang et. al, The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis, Proceedings of the Royal Society of London, 1998
        - S.E. Hamdi et al., Hilbert-Huang Transform versus Fourier based analysis for diffused ultrasonic waves structural health monitoring in polymer based composite materials, Proceedings of the Acoustics 2012 Nantes Conference.
        """
        includeRes = kwargs.get('includeRes', False)
        decimateTime = kwargs.get('decimateTime', False)
        filterSigma = kwargs.get('filterSigma', 0)

        # Get discrete matrix
        f, t, spectrumMat = self.discreteMatrix( frequencyBins=frequencyBins, includeRes=includeRes)

        # Decimate the time axis to a manageble length
        if decimateTime != False:
            spectrumMat = signal.decimate(spectrumMat, q=decimateTime, axis=1)
            t = np.linspace(t[0], t[-1], spectrumMat.shape[1])
        # Smooth the image for better intuision. 
        if 0<filterSigma:
            spectrumMat = ndimage.gaussian_filter(spectrumMat, sigma=filterSigma, mode='reflect')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cset = ax.pcolormesh(t, f, spectrumMat )
        plt.colorbar(cset, ax=ax)

        #ax.set_title("Hilbert Spectrum")
        ax.set_xlabel('$t$ [s]')
        ax.set_ylabel('$f$ [Hz]')
        ax.set_ylim(0,np.divide(self.Fs,2))
        ax.set_xlim(self.t[1], self.t[-1])
        ax.ticklabel_format(useMathText=True, scilimits=(0,3))
        plt.tight_layout()
        return fig, ax

    def spectrum( self, *args, **kwargs):
        """Hilbert-Huang transform with Hilbert spectral plot.
        The plot neglects negative frequencies.

        :return: [description]
        :rtype: [type]

        :Keyword Arguments:
        * *includeRes* (``bool``) -- Defines whether the residu is included in the spectrum, defaults to False

        - Huang et. al, The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis, Proceedings of the Royal Society of London, 1998
        - S.E. Hamdi et al., Hilbert-Huang Transform versus Fourier based analysis for diffused ultrasonic waves structural health monitoring in polymer based composite materials, Proceedings of the Acoustics 2012 Nantes Conference.
        """
        IncludeRes = kwargs.get('IncludeRes', False)

        # Calculate Hilbert spectrum

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cset = None
        for i in range(np.size(self.IF,0)):
            cset = ax.scatter(self.t, self.IF[i], c=self.intensity[i], s=5, alpha=0.3)
        plt.colorbar(cset, ax=ax)

        ax.set_title("Hilbert Spectrum")
        ax.set_xlabel('t [s]')
        ax.set_ylabel('f [Hz]')
        ax.set_ylim(0,np.divide(self.Fs,2))
        ax.set_xlim(self.t[1], self.t[-1])
        ax.ticklabel_format(useMathText=True, scilimits=(0,3))
        plt.tight_layout()
        return fig, ax


def FAM(x, *args, **kwargs):
    """Estimate the discrete time Spectral Correlation Density (SCD) using the Time-Smoothing FFT Accumulation Method.

    :param x: The time series being analyzed
    :type x: ndarray, time series
    :param \**kwargs:
            See below
    :return: SCD, f_j, alpha_i. A matrix of the specral correlation for frequencies f_j and cyclefrequencies alpha_i
    :rtype: ndarray

    :Keyword Arguments:
        * *plot* (``bool``) -- Defines whether to plot the SCD, defaults to False
        * *scale* (``str``) -- Defines the scaling, 'absolute', 'log', 'dB'
        * *Fs* (``scalar``) -- Sample frequency [Hz]
        * *method* (``str``) -- Defines the method, 'conj', 'non-conj'

    - Roberts et al., Computationally Efficient Algorithms for Cyclic Spectral Analysis, IEEE SP Magazine, 1991
    - C Spooner, CSP Estimators: The FFT Accumulation Method, https://cyclostationary.blog/2018/06/01/csp-estimators-the-fft-accumulation-method/, 2018
    """
    plot = kwargs.get('plot', False)
    scale = kwargs.get('scale', None) # 'absolute', 'log', 'dB'
    Fs = kwargs.get('Fs', 1)
    method = kwargs.get('method', 'conj') # 'non-conj' or conj

    # TODO: Add support for both x(t) and y(t)
    # L << N
    b = 8
    N = np.intc(2**b)
    L = np.intc(N/512)
    N_Prime = np.intc(L*1.8)
    P = np.intc(N/L)

    # Scale estimator to input signal
    while (P*L+N<len(x)):
        b += 1
        N = np.intc(2**b)
        L = np.intc(N/512)
        N_Prime = np.intc(L*1.8)
        P = np.intc(N/L)

    # Ensure input is complex
    x = x.astype(complex)

    # Zero padd signal in order fo fill N*P matrix
    if len(x)<(P-1)*L+N_Prime-len(x):
        x = np.pad(x, (0, (P-1)*L+N_Prime), 'constant', constant_values=(0, 0))

    # Assemble data matrix from input sequence
    xMat = np.zeros((N_Prime, P), dtype=complex)

    for p in range(np.size(xMat, 1)):
        xMat[:,p] = x[p*L:(p*L)+N_Prime]

    # Apply window of len N to the data columnwise
    window = signal.hamming(N_Prime)

    for p in range(np.size(xMat, 1)):
        #for column in xMatRRW.T:
            # Window column
            xMat[:,p] = np.multiply(xMat[:,p], window)
            # N-point FFT
            xMat[:,p] = np.fft.fft(xMat[:,p])
            # Correct phase
            nVec = np.array(range(1, N_Prime+1))
            xMat[:,p] = np.multiply(xMat[:,p],np.exp(np.multiply(-1j*2*np.pi*(p+1),nVec)/N_Prime))

    # Mix Channelized Subblocks (Self mix)
    SCD = np.empty_like(xMat, dtype=complex)

    for j in range(np.size(SCD, 0)):
        if method == 'non-conj':
            SCD[j,:] = np.multiply(xMat[j,:], np.conjugate(xMat[j,:]))
        elif method == 'conj':
            SCD[j,:] = np.multiply(xMat[j,:], xMat[j,:])
        # P-point FFT
        SCD[j,:] = np.fft.fft(SCD[j,:])

    # Shift data (zero hertz at the center of each axis)
    SCD = np.fft.fftshift(SCD)
    
    # Calculate axis
    k = np.linspace(-N_Prime/2, N_Prime/2-1, N_Prime)
    f_j = k*(Fs/N_Prime)    # Frequency axis
    
    deltaAlpha = Fs/N
    alpha_i = np.linspace(-np.size(SCD, 1)/2, (np.size(SCD, 1)/2)-1, np.size(SCD, 1))*deltaAlpha
    
    angSCD = np.angle(SCD)
    
    # Scale output
    if scale=='dB':
        SCD = util.pow2db(np.abs(SCD))
    elif scale=='log':
        SCD = np.log(np.abs(SCD))
    elif scale=='absolute':
        SCD = np.abs(SCD)

    # Plot SCD
    if plot == True:
        # Plot magnitude
        plt.figure()
        if scale=='linear':
            SCDplt = np.abs(SCD)
        else:
            SCDplt = SCD

        plt.pcolormesh(alpha_i, f_j, SCDplt)
        plt.title("Spectral Correlation Density")
        plt.xlabel("alpha [Hz]")
        plt.ylabel("f [Hz]")
        plt.colorbar()

        # Plot phase
        plt.figure()
        plt.pcolormesh(alpha_i, f_j, angSCD)
        plt.title("Spectral Correlation Density (Phase)")
        plt.xlabel("alpha [Hz]")
        plt.ylabel("f [Hz]")
        plt.colorbar()
    return SCD, f_j, alpha_i

def cyclicEstimator( SCD, f, alpha, bandLimited=True, **kwargs):
    """Estimates center frequency and symbol rate from a Spectral Correlation Density.

    :param SCD: An m,n matrix of the Spectral Correlation Density (complex).
    :type SCD: ndarray, matrix
    :param f: A frequency vector of length m
    :type f: ndarray, vector
    :param alpha: A cyclic frequency vector of length n
    :type alpha: ndarray, vector
    :param bandLimited: When set to True, the symbol tate estimator only utilizes the 1 dB bw for estimation. Estimator relies on a >1 dB inband SNR., defaults to True
    :type bandLimited: bool, optional
    :return: Estimated center frequency and symbol rate.
    :rtype: scalar
    """
    # Find row for alpha=0
    alpha0Index = np.argmin(np.abs(alpha))
    deltaAlpha = (alpha[-1]-alpha[0])/(len(alpha)-1)
    deltaFreq = (f[-1]-f[0])/(len(f)-1)
    # Estimate IF by use of maximum likelyhood estimation.
    # Multiply alpha dimension with triangle vector for improved center frequency estimation.
    triangleAlpha = signal.triang(len(alpha)) # Triangle window of length len(alpha).
    alphaWindow = np.multiply(np.ones(len(alpha)), triangleAlpha) # Triangular window
    #alphaWindow = np.ones(len(alpha)) Rectangular window
    alphaWindow[alpha0Index-2:alpha0Index+2] = 0
    # alphaWindow = alphaWindow / np.sum(alphaWindow) # Normalize
    # Allow externally configured window.
    alphaWindow = kwargs.get('alphaWindow', alphaWindow)
    
    freqEstVetctor = np.dot(alphaWindow, np.abs(SCD.T)) # Utilize the cyclic dimension for frequency estimation.
    fWindow = kwargs.get('fWindow', np.array([1]))

    # Allow parametric window creation.
    if isinstance(fWindow, str):
        # The width of the window in Hertz.
        winLenHertz = kwargs.get('fWindowWidthHertz', 50)
        winLen = np.intc(np.ceil(winLenHertz/deltaFreq))
        if fWindow=='triangle':
            fWindow = signal.triang(np.intc(winLen))#/winLen # Triangle window of length len(f).
        elif fWindow=='rectangle':
            # create rectangular window.
            fWindow = np.ones(winLen)
        # Filter along frequency axis.
        freqEstVetctor = signal.fftconvolve(freqEstVetctor, fWindow, mode='same')
    # If no filtering is intended.
    elif len(fWindow)!=1:
        # Filter along frequency axis.
        freqEstVetctor = signal.fftconvolve(freqEstVetctor, fWindow, mode='same')

    # Allow a-priori center frequency and bandwidth
    fCenter = kwargs.get('fCenterPriori', None)
    if (fCenter!=None) & (len(fWindow)!=1):
        fCenterIndex = np.argmin(np.abs(f-fCenter))
        fLowerIndex = np.intc(np.floor(fCenterIndex-(len(fWindow)/2)))
        fUpperIndex = np.intc(np.floor(fCenterIndex+(len(fWindow)/2)))
        alphaAverage = np.dot(np.abs(SCD[fLowerIndex:fUpperIndex, :].T), fWindow)
    # Estimate symbol rate through maximization of pulse train correlation.
    elif bandLimited == True:
        # Estimate signal bandwidth.
        fCenter, bw, fUpper, fLower, fCenterIndex, fUpperIndex, fLowerIndex = bandwidthEstimator(util.pow2db(freqEstVetctor), f, 2)
        bandWindow = np.ones(fUpperIndex-fLowerIndex)
        alphaAverage = np.dot(np.abs(SCD[fLowerIndex:fUpperIndex, :].T), bandWindow)
    else:
        fCenter = None
        alphaAverage = np.sum(np.abs(SCD), 0)

    # Take number of peaks in estimation as input
    symRateCyclicPeakNumber = kwargs.get('symRateCyclicPeakNumber', 5)
    R_symb = f0MLE(alphaAverage, alpha, symRateCyclicPeakNumber)
    return fCenter, R_symb


def f0MLE(psd, f, peaks, blankDistance=2, resolution=4): # Blank distance has been 4 for estimation of NLFM symbol rate
    """Maximum likelihood estimation of the fundamental frequency of a signal with repeating harmonics in the frequiency domain.
    A frequency domain intrepetation of [1].

    :param psd: the two-sided frequency domain representation of the signal under observation
    :type psd: ndarray, vector
    :param f: The frequency vector of PSD
    :type f: ndarray, vector
    :param peaks: The number of harmonic peaks to include in the estimation
    :type peaks: integer, scalar
    :param blankDistance: The distance which the estimats are neglected, defaults to 2
    :type blankDistance: int, optional
    :param resolution: The fractional resolution of the estimate. When resolution=1, then the PSD resolution is the bound., defaults to 4
    :type resolution: int, optional
    :return: Estimated funcamental frequency
    :rtype: scalar
    
    [1] Wise et. al, Maximum likelihood pitch estimation, IEEE Transactions on Acoustics, Speech, and Signal Processing, 1976
    """
    # Convert psd to singlesided
    psd = np.add( psd[np.intc(len(psd)/2):np.intc(len(psd)-1)], np.flip(psd[0:np.intc((len(psd)/2)-1)]) )
    f = f[np.intc(len(f)/2):len(f)-1]

    K = peaks
    k = np.linspace(1, K, K)

    fDelta = (f[-1]-f[0])/(len(f)-1)

    # Rough search
    f0Vec = np.linspace(blankDistance*fDelta, f[-1]/K, np.intc(len(f)-blankDistance))
    lossInv = np.zeros(len(f0Vec))
    for i, f0 in enumerate(f0Vec):
        f0Disc = f0/fDelta
        idx = np.intc(f0Disc*k)
        lossInv[i] = np.sum(psd[idx])

    f0 = f0Vec[np.argmax(lossInv)]

    # Fine search
    f0Vec = np.linspace(f0-fDelta, f0+fDelta, 2*resolution)
    lossInv = np.zeros(len(f0Vec))
    for i, f0 in enumerate(f0Vec):
        f0Disc = f0/fDelta
        idx = np.intc(f0Disc*k)
        while len(psd) <= idx[-1]:
            idx = idx[:-1]

        lossInv[i] = np.sum(psd[idx])
    f0 = f0Vec[np.argmax(lossInv)]
    return f0

def f0MleTime(Rxx, f, peaks, blankDistance=20, resolution=1):
    """Maximum likelihood estimation of the fundamental frequency of a signal with repeating autocorrelation peaks.

    :param Rxx: The two-sided ('full') autocorrelation
    :type Rxx: ndarray, vector
    :param f: The sampling frequency [Hz]
    :type f: scalar
    :param peaks: The number of harmonic peaks to include in the estimation
    :type peaks: int, scalar
    :param blankDistance: The distance from full overlap wchich is to be ignored in the estimation, defaults to 20
    :type blankDistance: int, optional
    :param resolution: [description], defaults to 1
    :type resolution: int, optional
    :return: The fundamental/cyclic frequency f0
    :rtype: scalar

    [1] Wise et. al, Maximum likelihood pitch estimation, IEEE Transactions on Acoustics, Speech, and Signal Processing, 1976
    """
    Fs = f
    dt = (1/Fs)
    # Singlesided RXX as per the definision in [1]
    Rxx = Rxx[np.intc(len(Rxx)/2):]
    t = np.linspace(0, len(Rxx)*dt,len(Rxx))

    K = peaks
    k = np.linspace(1, K, K)

    tDelta = (t[-1]-t[0])/(len(t)-1)

    t0Vec = np.linspace(blankDistance*tDelta, t[-1]/K, len(t)*resolution)

    lossInv = np.zeros(len(t0Vec))
    for i, t0 in enumerate(t0Vec):
        t0Disc = np.intc(t0/tDelta)
        idx = np.intc(t0Disc*k)
        lossInv[i] = np.sum(Rxx[idx])

    t0 = t0Vec[np.argmax(lossInv)]
    f0=1/t0
    return f0

def instFreq(sig_t, Fs, method='derivative', *args, **kwargs):
    """Estimate the instantaneous frequency of a time series.

    :param sig_t: Time series being analyzed
    :type sig_t: ndarray, time series
    :param Fs: The sample frequency [Hz]
    :type Fs: scalar
    :param method: Parameter deciding the estimator, defaults to 'derivative'

        - 'derivative' is a numerical approximation of the direct derivative approach
        - 'BarnesTwo' is Barnes "two-point filter approximation".
        - 'BarnesThree' is the Barnes "three-point filter approximation".
        - 'Claerbouts' is the Claerbouts approximation.
        - 'maxDHHT' is a numerical maximum likelihood method on the Discretized Hilbert spectrum.
        - 'polyLeastSquares' uses a method of phase polynomial with a least squares coefficient estimation. 
        - 'polyMle' uses a method of phase polynomial with a maxumum likelihood coefficient estimation.
    :type method: str, optional
    :return: The instantaneous frequency as a function of time
    :rtype: ndarray, time series
    
    - A. E. Barnes, The calculation of instantaneous frequency and instantaneous bandwidth, GEOPHYSICS, VOL. 57, NO. 11, 1992
    - Boashash et. al, Algorithms for instantaneous frequency estimation: a comparative study, Proceedings of SPIE, 1990
    """

    if np.isrealobj(sig_t):
        sig_t = signal.hilbert(sig_t)

    def Derivative(sig_t, Fs):
        phi_t = np.unwrap(np.angle(sig_t))
        omega_t = np.gradient(phi_t, 1/Fs)
        f_t = np.divide(omega_t, 2*np.pi)
        return f_t

    def BarnesTwo(sig_t, Fs):
        T=1/Fs
        x = np.real(sig_t)
        y = np.imag(sig_t)

        a = np.multiply(x[:-1], y[1:])
        b = np.multiply(x[1:], y[:-1])
        c = np.multiply(x[:-1], x[1:])
        d = np.multiply(y[:-1], y[1:])
        f_t = 1/(2*np.pi*T)*np.arctan( np.divide(np.subtract(a, b), np.add(c, d)) )
        f_t = np.append(f_t, f_t[-1])
        return f_t

    def BarnesThree(sig_t, Fs):
        T=1/Fs
        x = np.real(sig_t)
        y = np.imag(sig_t)

        a = np.multiply(x[:-2], y[2:])
        b = np.multiply(x[2:], y[:-2])
        c = np.multiply(x[:-2], x[2:])
        d = np.multiply(y[:-2], y[2:])
        f_t = 1/(4*np.pi*T)*np.arctan( np.divide(np.subtract(a, b), np.add(c, d)) )
        return f_t

    def Claerbouts(sig_t, Fs):
        T=1/Fs
        x = np.real(sig_t)
        y = np.imag(sig_t)

        a = np.multiply(x[:-1], y[1:])
        b = np.multiply(x[1:], y[:-1])
        c = np.multiply(x[:-1], x[1:])
        d = np.multiply(y[:-1], y[1:])
        f_t = 2/(np.pi*T)*( np.divide(np.subtract(a, b), np.add(np.power(c,2), np.power(d,2))) )
        return f_t

    def maxWVD(sig_t, Fs):
        tfr = tftb.processing.WignerVilleDistribution(sig_t)
        timeFreqMat, t, f = tfr.run()
        #tfr.plot(kind='contour', show_tf=True)
        f_t = Fs*f[np.argmax(timeFreqMat,0)]
        return f_t

    def maxDHHT(sig_t, Fs):
        HH = HilberHuang(np.real(sig_t), Fs)
        f, t, spectrumMat = HH.discreteMatrix(frequencyBins=256)

        f_t = np.empty(len(sig_t))

        for n, column in enumerate(spectrumMat.T):
            f_t[n] = f[np.argmax(column)]
        return f_t

    def polyLeastSquares(sig_t, Fs, order=6):
        T = len(sig_t)/Fs
        t = np.linspace(-T/2, T/2, len(sig_t))
        f_t = Derivative(sig_t, Fs)
        
        # Decimate to 1000 points
        dFactor = np.intc(len(sig_t)/1000)

        # Resample time series to improve the fitting result.
        fFit = signal.decimate(f_t, dFactor, ftype='iir', zero_phase=True)
        timeFit = np.linspace(-T/2, T/2, len(fFit))
                
        LsPoly = np.polyfit(timeFit, fFit, order)
        f_t = poly.polyval(t, LsPoly)
        return f_t

    def polyMle(sig_t, Fs, order, *args, **kwargs):
        """Estimate the instantaneous frequency through the use of a polynimial phase function and MLE coefficient estimation.

        :param sig_t: The time series to estimate.
        :type sig_t: ndarray, time series
        :param Fs: The sample frequency.
        :type Fs: scalar
        :param order: The polynomial order
        :type order: int, scalar
        :return: Instantaneous frequency
        :rtype: ndarray, time series

        - Boashash et. al, Algorithms for instantaneous frequency estimation: a comparative study, Proceedings of SPIE, 1990
        """
        windowSize = kwargs.get('windowSize', None)

        class polyOptim:
            def __init__( self, Fs, sig_t, t, windowSize=None):
                """
                Fs is the intended sampling frequency [Hz]. Fs must be at last twice the highest frequency in the input PSD. If Fs < 2*max(f), then Fs = 2*max(f)
                """
                self.z_t = np.array(sig_t, dtype=complex)    # complex observation
                self.Fs = Fs
                self.t = t
                self.T = len(sig_t)/Fs

            def objectFunction(self, a):
                """
                Object function to maximize.
                alpha is the parameter vector. alpha=[A, a_0, a_1,..., a_P]
                """
                a0 = np.array([0])
                #a = alpha[1:]
                aVec = np.append(a0, a)
                polyvec = poly.polyval(self.t, aVec)
                #A = alpha[0]

                D_alpha = (1/self.T)*np.sum(np.multiply(self.z_t, np.exp(np.multiply(-1j, polyvec))))
                #L = 2*A*np.real(np.exp(-1j*alpha[1])*D_alpha)*np.power(A, 2)
                L = np.abs(np.power(D_alpha, 2))
                return -L   # Negative as L is to be maximized

            def optimize(self, order, method='dual_annealing'):
                if method == 'dual_annealing':
                    bounds = list(zip([-1e6]*(order), [1e6]*(order)))
                    phaseOpt = optimize.dual_annealing(self.objectFunction, bounds=bounds)
                elif method == 'Nelder-Mead':
                    alpha0 = np.random.rand(order)
                    phaseOpt = optimize.minimize(self.objectFunction, alpha0, method='Nelder-Mead')
                elif method == 'basinhopping':
                    alpha0 = np.random.rand(order)
                    minimizer_kwargs = {"method": "BFGS"}
                    phaseOpt = optimize.basinhopping(self.objectFunction, alpha0, minimizer_kwargs=minimizer_kwargs, niter=200)

                a0 = np.array([0])
                alpha_hat = np.append(a0, phaseOpt.x)
                phasePoly = poly.Polynomial(alpha_hat)
                # Differentiate
                freqPoly = phasePoly.deriv()
                # Calculate IF
                f_t = 1/(2*np.pi)*poly.polyval(self.t, freqPoly.coef)
                return f_t, freqPoly.coef
        
        if windowSize==None:
            T = len(sig_t)/Fs
            t = np.linspace(-T/2, T/2, len(sig_t))
            m_polyOptim = polyOptim(Fs, sig_t, t=t)
            f_t, coeff = m_polyOptim.optimize(order)
        else:
            # Estimate for each chunk of the time series
            rest = len(sig_t) % windowSize
            #sig_t = np.append(sig_t, np.zeros(rest))
            f_t = np.empty(len(sig_t))
            dt=1/Fs           

            # Estimate the remainder
            if 0<rest:
                sig_rest_t = sig_t[-rest:]
                sig_t = sig_t[:-rest]
                t = np.linspace(-rest*dt/2, rest*dt/2, rest)
                m_polyOptim = polyOptim(Fs, sig_rest_t, t)
                f_t[-rest:], coeff = m_polyOptim.optimize(order)
            
            t = np.linspace(-windowSize*dt/2, windowSize*dt/2, windowSize)
            # Create matrix for iteration
            sigMat = np.reshape(sig_t, ( np.intc(len(sig_t)/windowSize), windowSize ))

            # Estimate in chunks
            for m, window in enumerate(sigMat):
                m_polyOptim = polyOptim(Fs, window, t)
                f_t[m*windowSize:(m+1)*windowSize], coeff = m_polyOptim.optimize(order, method='dual_annealing')
        return f_t


    if method=='derivative':
        f_t = Derivative(sig_t, Fs)
    elif method=='BarnesTwo':
        f_t = BarnesTwo(sig_t, Fs)
    elif method=='BarnesThree':
        f_t = BarnesThree(sig_t, Fs)
    elif method=='Claerbouts':
        f_t = Claerbouts(sig_t, Fs)
    elif method=='maxWVD':
        f_t = maxWVD(sig_t, Fs)
    elif method=='maxDHHT':
        f_t = maxDHHT(sig_t, Fs)
    elif method=='polyLeastSquares':
        order = kwargs.get('order', 4)
        f_t = polyLeastSquares(sig_t, Fs=Fs, order=order)
    elif method=='polyMle':
        order = kwargs.get('order', 4)
        windowSize = kwargs.get('windowSize', None)
        f_t = polyMle(sig_t, Fs=Fs, order=order, windowSize=windowSize)
    else:
        f_t = np.zeros(len(sig_t))
    return f_t


def carierFrequencyEstimator( sig_t, Fs, *args, **kwargs ):
    """Estimate the carrier frequency of a signal using an autocorrelation method, or a frequency domain maximum likelihood method.
    Autocorrelation method is applicable for sigle carrier signals as ASK, PSK, QAM.
    MLE method is applicable for the signal above in addition to continious carrier signals such as chirp.
    The MLE method utilizes either periodogram or Welch's method of spectral estimation.

    :param sig_t: The signal being analyzed.
    :type sig_t: ndarray, time series
    :param Fs: Sample frequency
    :type Fs: scalar
    :param \**kwargs:
            See below
    :return: Estimated center frequency
    :rtype: scalar

    :Keyword Arguments:
        * *method* (str) -- Decides which method is used, 'xcor' for autocorrelation method (default), 'mle' for maximum likelihood method.
        * *nfft* (int, scalar) -- Configures the length of the FFT used in the MLE method.

    Correlation method:

    - Z. Yu et al., A blind carrier frequency estimation algorithm for digitally modulated signals, IEEE 2004
    - Wang et al., Improved Carrier Frequency Estimation Based on Autocorrelation, Advances in Computer, Communication, Control and Automation, Springer 2011

    Maximum likelihood method:

    - Stotica et al., Maximum Likelihood Estimation of the Parameters of Multiple Sinusoids from Noisy Measurements, IEEE 1989
    """
    method = kwargs.get('method', None)
    nfft = kwargs.get('nfft', 2048)

    if method == None:
        method = 'xcor'
    
    if method == 'xcor':
        def autocorr(x):
            result = np.divide( signal.correlate(x, x, mode='full', method='fft'), len(x)-1 )
            return result[np.intc(len(result)/2):]

        L = len(sig_t)
        r_xx_l = autocorr(sig_t)
        Beta_l = np.angle(np.power(r_xx_l[1:] + np.conj(r_xx_l[:len(r_xx_l)-1]), 2))
        fCenter = Fs/(4*np.pi*(L-2))*np.sum(Beta_l)
    elif method == 'mle':
        # Select appropriate transform, periodogram or Welch's method
        if nfft<len(sig_t):
            f, p_xx = signal.welch(sig_t, Fs, nperseg=nfft, return_onesided=False) #! return_onesided=True 
            sig_f = np.sqrt(p_xx)
        else:
            f, sig_f = util.magnitudeSpectrum(sig_t, Fs, nfft=nfft)
        fCenter, fCenterIndex = fftQuadraticInterpolation(np.abs(sig_f), f)
    else:
        fCenter = None
    return fCenter

def fftQuadraticInterpolation(X_f, f, **kwargs):
    """Estimate the carrier frequency of a signal using quadratic interpolation.
    This function is called from carierFrequencyEstimator.

    :param X_f: The complex frequency domain vector of the signal.
    :type X_f: ndarray, vector
    :param f: The corresponding frequency vector.
    :type f: ndarray, vector
    :return: The center frequency and its index
    :rtype: scalar
    """
    # Magnitude vector
    mag = np.abs(X_f)

    # Find frequncy bin with highest magnitude
    k = np.argmax(mag)
    if k == 0:
        k=1
    elif k == (len(mag)-1):
        len(mag)-2

    #Quadratic fit around argmax and neighboring bins
    [a0, a1, a2] = np.polynomial.polynomial.polyfit(f[k-1:k+2], mag[k-1:k+2], 2)   

    # Fint Maximum
    fQuad = -a1/(2*a2)
    #t0 = T*n0
    #phase = np.angle(np.exp(-1j*2*np.pi*fQuad*t0)*X_f[k])
    return fQuad, k

def bandwidthEstimator( psd, f, threshold ):
    """Estimate the bandwidth of an incomming signal in the frequency domain.

    :param psd: The frequency domain signal to be analyzed (dB scale)
    :type psd: ndarray, vector
    :param f: The corresponding frequency vector
    :type f: ndarray, vector
    :param threshold: The power rollof at which the bandwidth is defined in dB
    :type threshold: scalar
    :return: fCenter, bw, fUpper, fLower, fCenterIndex, fUpperIndex, fLowerIndex
    :rtype: set of scalars
    """    
    fDelta = (f[-1]-f[0])/(len(f)-1)

    fCenter, fCenterIndex = fftQuadraticInterpolation(np.sqrt(util.db2pow(psd)), f)
    peakPowerdB = psd[fCenterIndex]
    thresholddB = peakPowerdB-threshold

    # fUpper
    fUpperIndex = fCenterIndex + np.argmin(np.abs(psd[fCenterIndex:]-thresholddB))
    fUpper = f[fUpperIndex]

    # fLower
    fLowerIndex = np.argmin(np.abs(psd[:fCenterIndex]-thresholddB))
    fLower = f[fLowerIndex]
    bw = fUpper - fLower
    return fCenter, bw, fUpper, fLower, fCenterIndex, fUpperIndex, fLowerIndex


def inspectPackage( sig_t, Fs, T, **kwargs ):
    """Estimate the number of unique symbols (pulses) in a packet

    :param sig_t: The extracted packet in time domain
    :type sig_t: time series
    :param Fs: The sampling frequency [Hz]
    :type Fs: scalar
    :param T: The symbol periond [s]
    :type T: scalar
    :return: The packet as a vector of symbols. Symbols are numbered as they appear in the signal.
    symbolAlpahebt, a vector of the time domain symbols.
    :rtype: set of two vectors

    :Keyword Arguments:
        * *threshold* (scalar) -- The threshhold for which two sybls are deemed identical
    """
    threshold = kwargs.get('threshold', 0.75)

    # Divide signal into symbols
    sampPerSymb = np.intc(T*Fs)
    
    # TODO: Align packet based on first symbol
    remainder = len(sig_t) % sampPerSymb
    if 0<remainder:
        sig_t = np.append(sig_t, np.zeros(remainder)) 
    nSymbols = np.intc(len(sig_t)/sampPerSymb)

    sigMat = sig_t.reshape((nSymbols,sampPerSymb))
    sigMat = sigMat.T

    # Correlate the first symbol with all other 
    extractionCount = np.ones(nSymbols)
    packet = np.zeros(nSymbols, dtype=int)

    iterator = 0
    symbolCount = 0
    while np.sum(extractionCount)!=0:
        if extractionCount[iterator]==1:
            correlationVector = np.zeros(nSymbols)
            for index, item in enumerate(extractionCount):
                if (item==1):
                    correlationVector[index] = np.max( np.abs(signal.correlate( sigMat[:,iterator], sigMat[:,index] , mode='same', method='fft')) )
            
            correlationVector = np.greater( correlationVector, correlationVector[iterator]*threshold )  # Compare with autocorrelation
            correlationVector[iterator] = True

            for index, item in enumerate(correlationVector):
                if item==True:
                    packet[index] = np.intc(symbolCount)
                    extractionCount[index] = 0

            symbolCount += 1
        iterator += 1

    # Output matrix
    symbolAlphabet = np.zeros((sampPerSymb,symbolCount), dtype=complex)
    symbNumb = 0
    for symbol in range(0,symbolCount):
        # Average the symbols
        for index, item in enumerate(packet):
            if item == symbNumb:
                symbolAlphabet[:,symbol] += sigMat[:,index]
        symbNumb += 1


        # Normalize power
        symbolAlphabet[:,symbol] = symbolAlphabet[:,symbol]/symbNumb

    return packet, symbolAlphabet
    