import scipy.signal as signal
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.special as special
import numpy as np
import numpy.polynomial.polynomial as poly

from mpl_toolkits.mplot3d import axes3d     # 3D plot
import matplotlib.pyplot as plt
from matplotlib import cm
colorMap = cm.coolwarm

from pyhht.visualization import plot_imfs   # Hilbert-Huang TF analysis
from pyhht import EMD                       # Hilbert-Huang TF analysis
import rftool.utility as util


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

def pulseCarrierCRLB(p_n, K, l_k, N):
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
    B0 = util.energy(np.gradient(p_n)) / E0
    # Time-frequency cross-coupling (skew)
    C0 = np.imag( np.sum( np.multiply( p_n, np.conj(np.gradient(p_n)) ) ) )
    R1 = np.mean(l_k[1:])
    R2 = np.mean(np.power(l_k[1:], 2))
    # Variance bound of frequency estimate
    var_theta = np.divide( N, 2*K*(E0-(np.power(C0, 2) / (B0*E0)))*(R2-np.power(R1, 2)) )
    return var_theta

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

def hilbert_spectrum( sig, Fs=1, *args, **kwargs):
    """
    Hilbert-Huang transform with Hilbert spectral plot.
    The plot neglects negative frequencies.
    
    sig is a time series
    Fs is the sample frequency

    Based on:
    - S.E. Hamdi et al., Hilbert-Huang Transform versus Fourier based analysis for diffused ultrasonic waves structural health monitoring in polymer based composite materials, Proceedings of the Acoustics 2012 Nantes Conference.
    """
    plotLabel = kwargs.get('label', None)

    # Hilbert-Huang
    decomposer = EMD(sig)
    imfs = decomposer.decompose()

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
    plt.figure(figsize=(10, 3))
    for i in range(np.size(instFreq,0)):
        plt.scatter(t, instFreq[i], c=intensity[i], s=5, alpha=0.3, cmap=colorMap)
    plt.legend(plotLabel)
    plt.colorbar()

    plt.title("Hilbert Spectrum")
    plt.xlabel('t [s]')
    plt.ylabel('f [Hz]')
    plt.ylim(0,np.divide(Fs,2))
    plt.xlim(t[1], t[-1])
    plt.tight_layout()

def ACF(x, plot = True, *args, **kwargs):
    """
    Normalized autocorrelation Function of input x.
    x is the signal being analyzed. If x is a matrix, the correlation is performed columnwise.
    plot decides wether to plot the result.
    label is the plot label for each vector. ['label1', 'label1']

    The output is 2*len(x)-1 long. If the input is a matrix, the output is a matrix.
    """
    plotLabel = kwargs.get('label', None)

    # If x is a vector, ensure it is a column vector.
    if x.ndim < 2:
        x = np.expand_dims(x, axis=1)

    # Iterate through columns
    r_xx = np.empty( shape=[2*np.size(x,0)-1, np.size(x,1)], dtype=complex )
    for n in range(0, np.size(x,1)):
        r_xx[:,n] = np.correlate(x[:,n], x[:,n], mode='full')
        
    if plot == True:
        # Plot
        plt.figure(figsize=(10, 3))
        
        for i, column in enumerate(r_xx.T):
            # Normalize
            column = np.absolute(column / abs(column).max())
            plt.plot(util.mag2db(column), label=plotLabel[i])

        yMin = np.maximum(-100, np.min(r_xx))
        plt.legend()
        plt.ylim([yMin, 0])
        plt.title("Autocorrelation Function")
        plt.ylabel("Normalized magnitude [dB]")
        plt.tight_layout()
    return r_xx

def FAM(x, *args, **kwargs):
    """
    Estimate the discrete time Spectral Correlation Density (SCD) using the Time-Smoothing FFT Accumulation Method.
    - Roberts et al., Computationally Efficient Algorithms for Cyclic Spectral Analysis, IEEE SP Magazine, 1991
    - C Spooner, CSP Estimators: The FFT Accumulation Method, https://cyclostationary.blog/2018/06/01/csp-estimators-the-fft-accumulation-method/, 2018
    """
    plot = kwargs.get('plot', False)
    scale = kwargs.get('scale', None) # 'absolute', 'log', 'dB'
    Fs = kwargs.get('Fs', 1)
    method = kwargs.get('method', 'non-conj') # 'non-conj' or conj

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
            xMat[:,p] = np.multiply(xMat[:,p],np.exp(np.multiply(-1j*2*np.pi*(p+1),nVec)/N_Prime))  # TODO, add alpha term

    # Mix Channelized Subblocks (Self mix)
    SCD = np.empty_like(xMat, dtype=complex)

    for j in range(np.size(SCD, 0)):
        if method == 'conj':
            SCD[j,:] = np.multiply(xMat[j,:], np.conjugate(xMat[j,:]))
        elif method == 'non-conj':
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

        plt.pcolormesh(alpha_i, f_j, SCDplt, cmap=colorMap)
        plt.title("Spectral Correlation Density")
        plt.xlabel("alpha [Hz]")
        plt.ylabel("f [Hz]")
        plt.colorbar()

        # Plot phase
        plt.figure()
        plt.pcolormesh(alpha_i, f_j, angSCD, cmap=colorMap)
        plt.title("Spectral Correlation Density (Phase)")
        plt.xlabel("alpha [Hz]")
        plt.ylabel("f [Hz]")
        plt.colorbar()
        
        # Plot Correlation Density as Surf
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #ax.set_zlim3d(0, np.max(SCD))
        Alpha_i, F_j = np.meshgrid(alpha_i, f_j)

        surf = ax.plot_surface(Alpha_i, F_j, SCDplt, cmap=colorMap, linewidth=0, antialiased=False)
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf) #, shrink=0.5, aspect=5)
        plt.title("Spectral Correlation Density")
        plt.xlabel("alpha [Hz]")
        plt.ylabel("f [Hz]")
        """

    return SCD, f_j, alpha_i

def f0MLE(psd, f, peaks):
    """
    Maximum likelihood estimation of the fundamental frequency of a signal with repeating harmonics in the frequiency domain.

    PSD is the two-sided frequency domain representation of the signal under observation.
    f is the frequency vector of PSD
    peaks, is the number of harmonic peaks to include in the estimation.

    - Wise et al., Maximum likelihood pitch estimation, IEEE Transactions on Acoustics, Speech, and Signal Processing, 1976
    """

    # Convert psd to singlesided
    psd = np.add( psd[np.intc(len(psd)/2):np.intc(len(psd)-1)], np.flip(psd[0:np.intc((len(psd)/2)-1)]) )
    f = f[np.intc(len(f)/2):len(f)-1]


    K = peaks
    k = np.linspace(1, K, K)

    fDelta = (f[-1]-f[0])/(len(f)-1)

    f0Vec = np.linspace(10, f[-1]/K, len(f))

    lossInv = np.zeros(len(f0Vec))
    for i, f0 in enumerate(f0Vec):
        f0Disc = f0/fDelta
        idx = np.intc(f0Disc*k)
        lossInv[i] = np.sum(psd[idx])
    f0 = f0Vec[np.argmax(lossInv)]
    return f0

def instFreq(sig_t, Fs, method='derivative', *args, **kwargs):
    """
    Estimate the instantaneous frequency of a time series.

    sig_t is the signal time series.
    Fs is the sample frequency.
    method decides the estimation method:
        'derivative' is a numerical approximation of the following $f(t) = \frac{1}{2\pi}\od{\Phi(t)}{t}$
        'BarnesTwo' is Barnes "two-point filter approximation".
        'BarnesThree' is the Barnes "three-point filter approximation".
        'Claerbouts' is the Claerbouts approximation.
        'polyMle' uses a method of phase polynomial.

    Returns the instantaneous frequency over time.
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
    
    def polyMle(sig_t, Fs, order=3):
        """
        Estimate the instantaneous frequency through the use of a polynimial phase function and MLE coefficient estimation.

        sig_t is the time series to estimate.
        Fs is the sample frequency.
        order is the polynomial order.

        - Boashash et. al, Algorithms for instantaneous frequency estimation: a comparative study, Proceedings of SPIE, 1990
        """
        class polyOptim:
            def __init__( self, Fs, sig_t ):
                """
                Fs is the intended sampling frequency [Hz]. Fs must be at last twice the highest frequency in the input PSD. If Fs < 2*max(f), then Fs = 2*max(f)
                """
                self.z_t = np.array(sig_t, dtype=complex)    # complex observation
                self.Fs = Fs
                self.T = len(sig_t)/Fs
                self.t = np.linspace(-self.T/2, self.T/2, len(sig_t))

            def objectFunction(self, alpha):
                """
                Object function to maximize.
                alpha is the parameter vector. alpha=[A, a_0, a_1,..., a_P]
                """
                a0 = np.array([0])
                a = alpha[2:]
                aVec = np.append(a0, a)
                polyvec = poly.polyval(self.t, aVec)
                A = alpha[0]

                D_alpha = (1/self.T)*np.sum(np.multiply(self.z_t, np.exp(np.multiply(-1j, polyvec))))
                L = 2*A*np.real(np.exp(-1j*alpha[1])*D_alpha)*np.power(A, 2)
                return -L

            def optimize(self, order):
                alpha0 = np.random.rand(order+2)
                #phaseOpt = optimize.minimize(self.objectFunction, alpha0, method='Nelder-Mead')

                minimizer_kwargs = {"method": "BFGS"}
                phaseOpt = optimize.basinhopping(self.objectFunction, alpha0, minimizer_kwargs=minimizer_kwargs, niter=200)
                
                alpha_hat = phaseOpt.x
                phasePoly = poly.Polynomial(alpha_hat[1:])
                # Differentiate
                phasePoly = phasePoly.deriv()
                # Calculate IF
                f_t = 1/(2*np.pi)*poly.polyval(self.t, phasePoly.coef)
                return f_t

        m_polyOptim = polyOptim(Fs, sig_t)
        f_t = m_polyOptim.optimize(order)
        return f_t

    if method=='derivative':
        f_t = Derivative(sig_t, Fs)
    elif method=='BarnesTwo':
        f_t = BarnesTwo(sig_t, Fs)
    elif method=='BarnesThree':
        f_t = BarnesThree(sig_t, Fs)
    elif method=='Claerbouts':
        f_t = Claerbouts(sig_t, Fs)
    elif method=='polyMle':
        f_t = polyMle(sig_t, Fs)
    return f_t


def carierFrequencyEstimator(sig_t, Fs, *args, **kwargs):
    """
    Estimate the carrier frequency of a signal using an autocorrelation method, or a frequency domain maximum likelihood method.
    Autocorrelation method is applicable for sigle carrier signals as ASK, PSK, QAM.
    MLE method is applicable for the signal above in addition to continious carrier signals such as chirp.
        The MLE method utilizes either periodogram or Welch's method of spectral estimation.

    sig_t is the signal being analyzed.
    Fs is the sampling frequency.
    method decides which method is used, 'xcor' for autocorrelation method (default), 'mle' for maximum likelihood method.
    nfft configures the length of the FFT used in the MLE method.

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
            f, p_xx = signal.welch(sig_t, Fs, nperseg=nfft, return_onesided=True)
            sig_f = np.sqrt(p_xx)
        else:
            f, sig_f = util.magnitudeSpectrum(sig_t, Fs, nfft=nfft)
        fCenter, fCenterIndex = fftQuadraticInterpolation(np.abs(sig_f), f)
    else:
        fCenter = None

    return fCenter

def fftQuadraticInterpolation(X_f, f):
    """
    Estimate the carrier frequency of a signal using quadratic interpolation.
    This function is called from carierFrequencyEstimator.

    X_f is the complex frequency domain vector of the signal.
    f is the corresponding frequency vector.

    Returns the center frequency and its index
    """
    # Magnitude vector
    mag = np.abs(X_f)

    # Find frequncy bin with highest magnitude
    k = np.argmax(mag)

    #Quadratic fit around argmax and neighboring bins
    [a0, a1, a2] = np.polynomial.polynomial.polyfit(f[k-1:k+2], mag[k-1:k+2], 2)

    """
    #! Debug code
    plt.figure()
    plt.stem(f[k-1:k+2], mag[k-1:k+2])
    polyFreq = np.linspace(f[k-1], f[k+1], 11)
    polyCurve = np.polynomial.polynomial.polyval(polyFreq, [a0, a1, a2])
    plt.plot( polyFreq, polyCurve )
    plt.show()
    #! End debug code
    """

    # Fint Maximum
    fQuad = -a1/(2*a2)
    #t0 = T*n0
    #phase = np.angle(np.exp(-1j*2*np.pi*fQuad*t0)*X_f[k])
    return fQuad, k

def bandwidthEstimator(psd, f, threshold):
    """
    Estimate the bandwidth of an incomming signal in the frequency domain.

    psd is the frequency domain signal to be analyzed (dB scale).
    xAxis is the time or frequency axis.
    threshold is the power rollof at which the bandwidth is defined in dB.

    Returns center frequency and threshold dB bandwidth.
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


def cyclicEstimator( SCD, f, alpha, bandLimited=True ):
    """
    Estimates center frequency and symbol rate from a Spectral Correlation Density.
    SCD is an m,n matrix of the Spectral Correlation Density (complex).
    f is a frequency vector of length m.
    alpha is a cyclic frequency vector of length n.
    bandLimited. When set to True, the symbol tate estimator only utilizes the 1 dB bw for estimation. Estimator relies on a >1 dB inband SNR.

    returns estimated center frequency and symbol rate.
    """
    # Find row for alpha=0
    alpha0Index = np.argmin(np.abs(alpha))
    deltaAlpha = (alpha[-1]-alpha[0])/(len(alpha)-1)
    # Estimate IF by use of maximum likelyhood estimation.
    # Multiply alpha dimension with triangle vector for improved center frequency estimation.
    triangleAlpha = signal.triang(len(alpha)) # Triangle window of length len(alpha).
    window = np.multiply(np.ones(len(alpha)), triangleAlpha)
    window[alpha0Index-2:alpha0Index+2] = 0
    window = window / np.sum(window)
    freqEstVetctor = np.dot(window, np.abs(SCD.T))       # Utilize the cyclic dimension for frequency estimation.
    triLen = len(f)/30
    triangleF = signal.triang(np.intc(triLen))/triLen # Triangle window of length len(f).
    filteredFreqEstVetctor = signal.fftconvolve(freqEstVetctor, triangleF, mode='same') # TODO Check if there are zero-values in freqEstVetctor prior to the filtering.

    # Estimate symbol rate through maximization of pulse train correlation
    if bandLimited == True:
        # Estimate signal bandwidth
        fCenter, bw, fUpper, fLower, fCenterIndex, fUpperIndex, fLowerIndex = bandwidthEstimator(util.pow2db(filteredFreqEstVetctor), f, 2)
        bandWindow = np.ones(fUpperIndex-fLowerIndex)
        alphaAverage = np.dot(np.abs(SCD[fLowerIndex:fUpperIndex, :].T), bandWindow)
    else:
        fCenter = None
        alphaAverage = np.sum(SCD, 0)

    R_symb = f0MLE(alphaAverage, alpha, 5) #! Debug, restet to 6
    return fCenter, R_symb


class chirp:
    """
    Object for generating linear and non-linear chirp generation.
    """
    t = None
    T = None

    def __init__( self, Fs=1 ):
        """
        t_i is the chirp duration [s].
        Fs is the intended sampling frequency [Hz]. Fs must be at last twice the highest frequency in the input PSD. If Fs < 2*max(f), then Fs = 2*max(f)
        """
        self.Fs = Fs
        self.dt = 1/self.Fs

    def checkSampleRate(self):
        """
        Check that the sample rate is within the nyquist criterion. I.e . that no phase difference between two consecutive samples exceeds pi.
        """
        errorVecotr = np.abs(np.gradient(self.targetOmega_t)*2*np.pi)-np.pi

        if 0 < np.max(errorVecotr):
            print("Warning, sample rate too low. Maximum phase change is", np.pi+np.max(errorVecotr), "Maximum allowed is pi." )
    
    def genFromPoly( self, direction = None ):
        """
        Generate Non-Linear Frequency Modualted (NLFM) chirps based on a polynomial of arbitrary order.
        direction controls the chirp direction. 'inverted' inverts the chirp direction.
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
        """
        Generate Non.Linear Frequency Modualted (NLFM) chirps.
        direction controls the chirp direction. 'inverted' inverts the chirp direction.
        """
        dt = 1/self.Fs        # seconds

        omega_t = self.targetOmega_t
        if direction == 'inverted':
            omega_t = np.max(omega_t) - (omega_t-np.min(omega_t))

        phi_t = util.indefIntegration( omega_t, dt )
        sig = np.exp(np.multiply(1j*2*np.pi, phi_t))
        return sig

    def getInstFreq(self, poly=True, plot=True):
        # Calculate the instantaneous frequency as a function of time
        if poly == True:
            # Calculate the instantaneous frequency based on polynoimial coefficients.
            polyOmega = np.poly1d(self.c)
            omega_t = polyOmega(self.t)
        else:
            # Calculate the instantaneous frequency based on phase vector.
            omega_t = np.gradient(self.phi_t, self.t)

        if plot == True:
            plt.figure(figsize=(10, 3))
            plt.plot(self.t, omega_t)
            plt.plot(self.t, self.targetOmega_t)
            plt.xlabel('t [s]')
            plt.ylabel('f [Hz]')
            plt.title("Instantaneous Frequency")
            plt.show()
        return omega_t

    def getChirpRate(self, poly=True, plot=True):
        # Calculate the chirp rate as a function of time
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
            plt.figure(figsize=(10, 3))
            plt.plot(self.t, gamma_t)
            plt.xlabel('t [s]')
            plt.ylabel('f [Hz]')
            plt.title("Chirp Rate")
            plt.show()
        return gamma_t

    def PSD( self, sig_t, plot=False ):
        """
        Calculates Power Spectral Density in dBW/Hz.
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
        """
        Lookup table for the W function. Takes instantaneous frequency as input.
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
        """
        Objective function for finding gamma_t that meets the constraints.
        scale scales the gamma function.
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
        """
        Calculate the necessary coefficients in order to generate a NLFM chirp with a specific magnitude envelope (in frequency domain). Chirp generated using rftool.radar.generate().
        Coefficients are found through non-linear optimization.

        Window_ is the window function for the target PSD. It is used as a LUT based function from -Omega/2 to Omega/2, where Omega=targetBw.
        T is the pulse duration [s].
        targetBw is the taget bandwidth of the chirp [Hz].
        centerFreq is the center frequency of the chirp [Hz].
        order is the oder of the phase polynomial used to generate the chirp frequency characteristics [integer].
        pints is the number of points used t evaluate the chirp function. Not to be confused with the number of samples in the genrerated IF chirp.
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
        
        # Order and initial conditions
        c0 = np.zeros( order )
        
        # Resample time series to improve the fitting result.
        omegaFit = signal.decimate(self.targetOmega_t, 16, ftype='iir', zero_phase=True)
        timeFit = np.linspace(-self.T/2, self.T/2, len(omegaFit))

        self.c = np.polyfit(timeFit, omegaFit, order)
        return self.c


    def modulate( self, bitstream=np.array([1,0,1,0])):
        """
        Modulate bit stream to a chirp. One chirp per bit. A 1 is represented as a forward time chirp.
        A zero is represented as a time-reversed chirp.
        
        bitStream is the bitstream to be modulated (numpy array).
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
