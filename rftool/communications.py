from scipy.stats import norm
import numpy as np
import scipy.constants as const
import rftool.utility as util

def Q( x ):
    """The Q-function. (just a translation for readability).
    """
    return norm.sf(x)

def errorProbabilityBpsk( EbN0 ):
    """Probability of error in AWGN as a function of Eb/N0 for Binary Phase Shift Keying (BPSK).

    :param EbN0: The intended ratio
    :type EbN0: ndarray, scalar or vector
    :return: Error probability

    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995 
    """
    return Q(np.sqrt( 2*EbN0 ))

def errorProbabilityQpsk( EbN0 ):
    """Probability of error in AWGN as a function of Eb/N0 for Quadrature Phase Shift Keying (QPSK).

    :param EbN0: The intended ratio
    :type EbN0: ndarray, scalar or vector
    :return: Error probability.

    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    return errorProbabilityBpsk( EbN0 )

def errorProbabilityMPsk( EbN0, M ):
    """Probability of error in AWGN as a function of Eb/N0 for M-Ary Phase Shift Keying (M-PSK).

    :param EbN0: The intended ratio
    :type EbN0: ndarray, scalar or vector
    :return: Error probability.

    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    return 2*Q( np.sqrt( 2*EbN0*np.log(M) )*np.sin(np.divide(const.pi, M)) ) # Technically "less than or equal"

def errorProbabilityFsk( EbN0 ):
    """Probability of error in AWGN as a function of Eb/N0 for non-coherent Frequency Shift Keying (FSK).

    :param EbN0: The intended ratio
    :type EbN0: ndarray, scalar or vector
    :return: Error probability.

    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    return np.divide(1,2)*np.exp(-2*EbN0)

def errorProbabilityCoherentFsk( EbN0 ):
    """Probability of error in AWGN as a function of Eb/N0 for coherent Frequency Shift Keying (FSK).

    :param EbN0: The intended ratio
    :type EbN0: ndarray, scalar or vector
    :return: Error probability.

    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    return Q(np.sqrt( EbN0 ))

def errorProbabilityCoherentMFsk( EbN0, M ):
    """Probability of error in AWGN as a function of Eb/N0 for coherent M-ary Frequency Shift Keying (M-FSK).

    :param EbN0: The intended ratio
    :type EbN0: ndarray, scalar or vector
    :param M: The order of the modultion
    :type M: int, scalar
    :return: Error probability.

    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    return (1-M)*Q(np.sqrt( EbN0*np.log(M) )) # Technically "less than or equal"

def errorProbabilityGMSK( EbN0 ):
    """Probability of error in AWGN as a function of Eb/N0 and the 3-dB bandwidth bit-dutation product, BT = 0.25 for Gaussian Minimum Shift Keying (GMSK).

    :param EbN0: The intended ratio
    :type EbN0: ndarray, scalar or vector
    :return: Error probability.

    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    gamma = 0.68
    return Q(np.sqrt(2*gamma*EbN0))

def errorProbabilityQam( EbN0 , M ):
    """Probability of error in AWGN as a function of Eb/N0 with the minimum Eb and order M for Quadrature Amplitude Modulaion (QAM).

    :param EbN0: The intended ratio
    :type EbN0: ndarray, scalar or vector
    :param M: The order of the modultion
    :type M: int, scalar
    :return: Error probability.

    - T. S. Rappaport, Wireless Communications Principles and Practice, 2nd ed, Prentice Hall, 1995
    """
    return 4*(1-np.divide(1, np.sqrt(M)))*Q( np.sqrt( 2*EbN0 ) )

def EbN0toSNRdB(EbN0, M, Fs, Fsymb):
    """Calculte the necessary SNR in order to obtain a target Eb/N0.

    :param EbN0: The intended ratio
    :type EbN0: ndarray, scalar or vector
    :param M: The order of the modultion
    :type M: int, scalar 
    :param Fs: The sample rate of the signal.
    :param Fsymb: The symbol rate of the signal (pulse rate).
    :return: Necessary SNR for obtaining a Eb/N0 in Deci-Bell form.
    """
    return util.pow2db(np.multiply(util.db2pow(EbN0), Fsymb*np.log2(M)/Fs))