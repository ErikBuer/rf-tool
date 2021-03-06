import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def Gamma2VSWR( Gamma ):
    """Convert between reflection coefficient Gamma and Voltage Standing Wave Ratio (VSWR)

    :param Gamma: The reflection coefficient
    :type Gamma: complex scalar
    :return: VSVR
    :rtype: real scalar
    """
    return np.divide(1+abs(Gamma), 1-abs(Gamma))

def pow2normdb( pow ):
    """Conversion between linear power normalized logaithmic decibel scale       

    :param mag: The magnitude
    :type mag: scalar, vector or matrix
    :return: The converted dB value 
    :rtype: scalar, vector or matrix
    """
    vec = 10*np.log10( pow )
    return 10*np.log10( pow )-np.max(10*np.log10( pow ))

def mag2db( mag ):
    """Conversion between linear magnitude (voltage etc.) and logaithmic decibel scale

    :param mag: The magnitude
    :type mag: scalar, vector or matrix
    :return: The converted dB value
    :rtype: scalar, vector or matrix
    """
    return 20*np.log10( mag )

def db2mag( dB ):
    """Conversion between logaithmic decibel and linear scale

    :param dB: dB value
    :type dB: scalar, vector or matrix
    :return: The converted magnitude (linear scale)
    :rtype: scalar, vector or matrix
    """
    return np.power( 10, np.divide( dB, 20 ) )

def pow2db( power ):
    """Conversion between linear power (Watt etc.) and logaithmic decibel scale

    :param power: The input power value
    :type power: scalar, vector or matrix
    :return: The converted power value
    :rtype: scalar, vector or matrix
    """
    return 10*np.log10( power )

def db2pow( dB ):
    """Conversion between logaithmic decibel and linear scale

    :param dB: dB value
    :type dB: scalar, vector or matrix
    :return: Conerted power value (linear scale)
    :rtype: scalar, vector or matrix
    """
    return np.power( 10, np.divide( dB, 10) )

def powerdB( x ):
    """Calculate the average signal power in dBW.

    :param x: A time series
    :type x: ndarray, time series
    :return: The avereage power
    :rtype: scalar
    """
    return pow2db(np.mean(np.power(np.abs(x), 2)))

def energy( x ):
    """Signal energy operator

    :param x: Time series
    :type x: ndarray, time series
    :return: Signal energy
    :rtype: scalar
    """
    return np.sum(np.power(np.abs(x), 2))

def wgndB( x, dB ):
    """Apply white Gaussian noise of specified power in dBW to signal.

    :param x: Clean time series
    :type x: ndarray, time series
    :param dB: Target noise power level [dBW]
    :type dB: scalar
    :return: Noisy signal
    :rtype: ndarray, time series
    """
    # Apply complex noise to comples signals. Half power noise to each component.
    if np.iscomplexobj(x)==True:
        wRe = np.random.normal(scale=np.sqrt(db2pow(dB)/2), size=np.shape(x))
        wIm = np.random.normal(scale=np.sqrt(db2pow(dB)/2), size=np.shape(x))
        w = wRe + 1j*wIm
    else:
        w = np.random.normal(scale=np.sqrt(db2pow(dB)), size=np.shape(x))
    return np.add(x,w)

def wgnSnr( x, SNRdB ):
    """Add noise to obtain an intendet SNR.

    :param x: Clean time series
    :type x: ndarray, time series
    :param SNRdB: Target SNR in dB
    :type SNRdB: scalar
    :return: Noisy time series
    :rtype: ndarray, time series
    """
    power = powerdB(x)
    noisePowerdB = power - SNRdB
    return wgndB( x, noisePowerdB )

def periodogram(sig, Fs, nfft = 2048):
    """Calculate power spectral density using a Periodogram

    :param sig: Time series
    :type sig: ndarray, time series
    :param Fs: Sample frequency [Hz]
    :type Fs: scalar
    :param nfft: DFT length, defaults to 2048
    :type nfft: int, optional
    """
    if nfft < len(sig):
        nfft = len(sig)

    sig_f = np.fft.fft(sig,nfft)/nfft
    # Shift and normalize
    sig_f = np.power(np.abs(np.fft.fftshift(sig_f)), 2)
    # Remove infinitesimally small components
    sig_f = pow2db(np.maximum(sig_f, 1e-16))
    # Generate frequency axis
    f = np.linspace(-Fs/2, Fs/2, len(sig_f))
    # Plot
    plt.figure(figsize=(10, 3))
    plt.plot(f, sig_f)
    plt.xlim([-Fs/2, Fs/2])
    plt.title("Power Spectral Density")
    plt.ylabel("Power density [dBW/Hz]")
    plt.xlabel("Frequency [Hz]")
    plt.tight_layout()

def welch(x, Fs, nfft = 2048):
    """Wrapper for welch spectral power estimate.

    :param x: Input Time series
    :type x: ndarray, time series
    :param Fs: Sample frequency [Hz]
    :type Fs: scalar    
    :param nfft: Length of DFT, defaults to 2048
    :type nfft: int, optional
    :return: f, Pxx_den
    :rtype: Set of two vectors
    """
    f, Pxx_den = signal.welch(x, Fs, nperseg=nfft, return_onesided=False)
    # Remove infinitesimally small components
    sig_f = pow2db(np.maximum(Pxx_den, 1e-16))
    # Plot
    plt.figure(figsize=(10, 3))
    plt.plot(f, sig_f)
    plt.xlim([-Fs/2, Fs/2])
    plt.title("Welch Power Spectral Density Estimate")
    plt.ylabel("Power density [dBW/Hz]")
    plt.xlabel("Frequency [Hz]")
    plt.tight_layout()
    return f, Pxx_den

def magnitudeSpectrum(sig, Fs, nfft = 2048, plot = False):
    """Normalized magnitude spectrumPower spectral density
    sig is the signal to be analyzed
    Fs is the sampling frequency [Hz]
    nfft is the length of the FFT. Is set to the singal length if the nfft<len(sig)

    :param sig: Inpu time series
    :type sig: ndarray, time series
    :param Fs: Sample frequency [Hz]
    :type Fs: scalar
    :param nfft: Length of DFT, defaults to 2048
    :type nfft: int, optional
    :param plot: Decides wheter to plot the spectrum, defaults to False
    :type plot: bool, optional
    :return: f, sig_f
    :rtype: set of two vectors
    """
    if nfft < len(sig):
        nfft = len(sig)

    sig_f = np.fft.fft(sig,nfft)/nfft
    # Shift and normalize
    sig_f = np.abs(np.fft.fftshift(sig_f / abs(sig_f).max()))
    # Generate frequency axis
    f = np.linspace(-Fs/2, Fs/2, len(sig_f))
    # Plot
    if plot == True:
        # Remove infinitesimally small components
        sig_f = mag2db(np.maximum(sig_f, 1e-10))

        plt.figure(figsize=(10, 3))
        plt.plot(f, sig_f)
        plt.xlim([-Fs/2, Fs/2])
        plt.title("Frequency response")
        plt.ylabel("Normalized magnitude [dB]")
        plt.xlabel("Frequency [Hz]")
    return f, sig_f

def indefIntegration( x_t, dt ):
    """Indefinite-like numerical integration. Takes in a vector (function), returns the integrated vector(function).

    :param x_t: Input vector
    :type x_t: ndarray, vector
    :param dt: Time delta between samples
    :type dt: scalar    
    :return: Numerical integral
    :rtype: ndarray, vector
    """
    Sx_tdt = np.cumsum(x_t)*dt
    return Sx_tdt