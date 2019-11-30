import numpy as np

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