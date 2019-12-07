from scipy.stats import norm
import numpy as np
import scipy.constants as const

def Q( x ):
    """
    The Q-function. (just a translation for readability).
    """
    Q = norm.sf(x)
    return Q

def errorProbabilityBpsk( EbN0 ):
    """
    Probability of error in AWGN as a function of Eb/N0 for Binary Phase Shift Keying (BPSK).
    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995 
    """
    Pe = Q(np.sqrt( 2*EbN0 ))
    return Pe

def errorProbabilityQpsk( EbN0 ):
    """
    Probability of error in AWGN as a function of Eb/N0 for Quadrature Phase Shift Keying (QPSK).
    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    Pe = errorProbabilityBpsk( EbN0 )
    return Pe

def errorProbabilityMPsk( EbN0, M ):
    """
    Probability of error in AWGN as a function of Eb/N0 for M-Ary Phase Shift Keying (M-PSK).
    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    Pe = 2*Q( np.sqrt( 2*EbN0*np.log(M) )*np.sin(np.divide(const.pi, M)) ) # Technically "less than or equal"
    return Pe

def errorProbabilityFsk( EbN0 ):
    """
    Probability of error in AWGN as a function of Eb/N0 for non-coherent Frequency Shift Keying (FSK).
    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    Pe = np.divide(1,2)*np.exp(-2*EbN0)
    return Pe

def errorProbabilityCoherentFsk( EbN0 ):
    """
    Probability of error in AWGN as a function of Eb/N0 for coherent Frequency Shift Keying (FSK).
    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    Pe = Q(np.sqrt( EbN0 ))
    return Pe


def errorProbabilityCoherentMFsk( EbN0, M ):
    """
    Probability of error in AWGN as a function of Eb/N0 for coherent M-ary Frequency Shift Keying (M-FSK).
    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    Pe = (1-M)*Q(np.sqrt( EbN0*np.log(M) )) # Technically "less than or equal"
    return Pe

def errorProbabilityGMSK( EbN0 ):
    """
    Probability of error in AWGN as a function of Eb/N0 and the 3-dB bandwidth bit-dutation product, BT = 0.25 for Gaussian Minimum Shift Keying (GMSK).
    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    gamma = 0.68
    Pe = Q(np.sqrt(2*gamma*EbN0))
    return Pe

def errorProbabilityQam( EbN0 , M):
    """
    Probability of error in AWGN as a function of Eb/N0 with the minimum Eb and order M for Quadrature Amplitude Modulaion (QAM).
    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    Pe = 4*(1-np.divide(1, np.sqrt(M)))*Q( np.sqrt( 2*EbN0 ) )
    return Pe