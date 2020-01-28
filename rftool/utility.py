import numpy as np
import matplotlib.pyplot as plt

def Gamma2VSWR( Gamma ):
    """
    Reflection coefficient to Voltage Standing Wave Ratio conversion
    Gamma is the reflection coefficient.
    """
    VSWR = np.divide(1+abs(Gamma), 1-abs(Gamma))
    return VSWR

def mag2db( mag ):
    """
    Conversion between linear magnitude (voltage etc.) and logaithmic scale
    """
    dB = 20*np.log10( mag)
    return dB

def db2mag( dB ):
    """
    Conversion between logaithmic decibell and linear scale
    """
    mag = np.power( 10, np.divide( dB, 20) )
    return mag

def pow2db( power ):
    """
    Conversion between linear power (Watt etc.) and logaithmic scale
    """
    dB = 10*np.log10( power)
    return dB

def db2pow( dB ):
    """
    Conversion between logaithmic decibell and linear scale
    """
    power = np.power( 10, np.divide( dB, 10) )
    return power

def powerSpectrum(sig, Fs, nfft = 2048):
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
    plt.figure()
    plt.plot(f, sig_f)
    plt.xlim([-Fs/2, Fs/2])
    plt.title("Power Spectral Density")
    plt.ylabel("Power density [dB/Hz]")
    plt.xlabel("Frequency [Hz]")
    plt.show()

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
    plt.figure()
    plt.plot(f, sig_f)
    plt.xlim([-Fs/2, Fs/2])
    plt.title("Frequency response")
    plt.ylabel("Normalized magnitude [dB]")
    plt.xlabel("Frequency [Hz]")
    plt.show()

def indefIntegration( x_t, dt ):
        """
        Indefinite-like numerical integration. Takes in a vector (function), returns the integrated vector(function).
        """
        Sx_tdt = np.cumsum(x_t)*dt
        return Sx_tdt