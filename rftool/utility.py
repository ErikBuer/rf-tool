import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def Gamma2VSWR( Gamma ):
    """
    Reflection coefficient to Voltage Standing Wave Ratio conversion
    Gamma is the reflection coefficient.
    """
    return np.divide(1+abs(Gamma), 1-abs(Gamma))

def mag2db( mag ):
    """
    Conversion between linear magnitude (voltage etc.) and logaithmic decibel scale
    """
    return 20*np.log10( mag)

def db2mag( dB ):
    """
    Conversion between logaithmic decibel and linear scale
    """
    return np.power( 10, np.divide( dB, 20 ) )

def pow2db( power ):
    """
    Conversion between linear power (Watt etc.) and logaithmic decibel scale
    """
    return 10*np.log10( power )

def db2pow( dB ):
    """
    Conversion between logaithmic decibel and linear scale
    """
    return np.power( 10, np.divide( dB, 10) )

def powerdB( x ):
    """
    Calculate the average signal power in dBW.
    """
    return pow2db(np.mean(np.power(np.abs(x), 2)))

def energy( x ):
    """
    Calculate the signal energy.
    """
    return np.sum(np.power(np.abs(x), 2))

def wgndB( x, dB ):
    """
    Apply white Gaussian noise of specified power in dBW to signal.
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
    """
    Add noise to obtain an intendet SNR.
    x is the input array.
    SNRdB is the target SNR in dB
    """
    power = powerdB(x)
    noisePowerdB = power - SNRdB
    return wgndB( x, noisePowerdB )

def periodogram(sig, Fs, nfft = 2048):
    """
    Power spectral density
    sig is the signal to be analyzed
    Fs is the sampling frequency [Hz]
    nfft is the length of the FFT
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

def welch(x, Fs, nfft = 2048):
    """
    Wrapper for welch spectral power estimate.
    sig is the signal to be analyzed
    Fs is the sampling frequency [Hz]
    nfft is the length of the FFT
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
    return f, Pxx_den

def magnitudeSpectrum(sig, Fs, nfft = 2048):
    """
    Normalized magnitude spectrumPower spectral density
    sig is the signal to be analyzed
    Fs is the sampling frequency [Hz]
    nfft is the length of the FFT
    """
    if nfft < len(sig):
        nfft = len(sig)

    sig_f = np.fft.fft(sig,nfft)/nfft
    # Shift and normalize
    sig_f = np.abs(np.fft.fftshift(sig_f / abs(sig_f).max()))
    # Remove infinitesimally small components
    sig_f = mag2db(np.maximum(sig_f, 1e-10))
    # Generate frequency axis
    f = np.linspace(-Fs/2, Fs/2, len(sig_f))
    # Plot
    plt.figure(figsize=(10, 3))
    plt.plot(f, sig_f)
    plt.xlim([-Fs/2, Fs/2])
    plt.title("Frequency response")
    plt.ylabel("Normalized magnitude [dB]")
    plt.xlabel("Frequency [Hz]")

def indefIntegration( x_t, dt ):
    """
    Indefinite-like numerical integration. Takes in a vector (function), returns the integrated vector(function).
    """
    Sx_tdt = np.cumsum(x_t)*dt
    return Sx_tdt