import numpy as np
import scipy.constants as const
import mpmath as mp

def Gamma2VSWR( Gamma ):
    """
    Reflection coefficient to Voltage Standing Wave Ratio conversion
    Gamma is the reflection coefficient.
    """
    VSWR = np.divide(1+abs(Gamma), 1-abs(Gamma))
    return VSWR


def effectivePermittivityHJ( h, w, e_r ):
    """
    Calculate effective permittivity from Hammerstad-Jensen (simplified formula).

    h   strip height over dielectric
    w   strip width
    e_r is the relative permittivity of the dielectric
    - T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
    """
    e_eff = np.divide((e_r+1),2) + np.divide((e_r-1),2)*np.divide(1,np.sqrt(1+(12*np.divide(h,w))))
    return e_eff


def Z01HJ( h, w, e_r ):
    """
    Calculate Z01 from Hammerstad-Jensen (simplified formula).
    Impedance instrip with air dielectric

    h   strip height over dielectric
    w   strip width
    e_r is the relative permittivity of the dielectric
    - T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
    """
    e_eff = effectivePermittivityHJ( h, w, e_r)
    u = np.divide(w,h)
    F1 = 6+(2*np.pi-6)*np.exp(-np.power(np.divide(30.666,u), 0.7528))
    z_01 = 60*np.log( np.divide(F1, u) + np.sqrt(1+np.power(np.divide(2,u),2)) )
    return z_01


def microstripImpedanceHJ( h, w, e_r ):
    """
    Calculate Characteristic Impedance from Hammerstad-Jensen (simplified formula).
    
                       w
                   <-------->
         e_0       +--------+                   
                   |        |                   
    +--------------+--------+--------------+ ^  
    |                                      | |
    |    e_r                               | | h
    |                                      | |
    +--------------------------------------+ v

    h is the strip height over dielectric
    w is the strip width
    e_r is the relative permittivity of the dielectric
    - T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
    """
    e_eff = effectivePermittivityHJ( h, w, e_r)
    z_01 = Z01HJ(h, w, e_r)
    z_0 = np.divide(z_01,np.sqrt(e_eff))
    return z_0


def effectiveStripWidthHJ( h, w, t, e_r ):
    """
    Calculate effective width w_eff for a microstrip of finite thickness. Hammerstad and Jenson's method.
    This effective width can be used in microstripImpedanceHJ to take strip thickness into account.

                       w
                   <-------->
         e_0       +--------+                   ^
                   |        |                   | t
    +--------------+--------+--------------+ ^  v
    |                                      | |
    |    e_r                               | | h
    |                                      | |
    +--------------------------------------+ v

    t is the strip thickness
    h is the strip height over dielectric
    w is the strip width
    e_r is the relative permittivity of the dielectric
    - T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
    """
    delta_w_1 = np.divide(t*h, const.pi)*np.log( float(1 + np.divide( 4*const.e, t*np.power(mp.coth( np.sqrt( 6.517*np.divide(w,h) ) ),2) ) ))
    delta_w_r = np.divide(1,2)*( 1+np.divide( 1, mp.cosh(np.sqrt(e_r-1)) ) )*delta_w_1
    w_eff = w + delta_w_r
    return w_eff

def shieldedMicrostripImpedanceHJ( h, w, t, a, b, e_r ):
    """
    Calculate Characteristic Impedance of microstrip in a metallic enclosure.
    Hammerstad-Jensen (simplified formula) quasi-static impedance.
    Using Hammerstad-Jensen effective width calculation.

    +--------------------------------------+ ^
    |                  w                   | |
    |              <-------->              | | a
    |    e_0       +--------+              | |   ^
    |              |        |              | |   | t
    +--------------+--------+--------------+ | ^ V
    |                                      | | |
    |    e_r                               | | | h
    |                                      | | |
    +--------------------------------------+ v v
    <-------------------------------------->
                   b
    
    h   strip height over dielectric
    w   strip width
    t is the strip thickness
    a is the enclosure height
    b is the enclosure width
    e_r is the relative permittivity of the dielectric
    - T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
    """
    z_0u = microstripImpedanceHJ( h, w, e_r )
    w_eff = effectiveStripWidthHJ( h, w, t, e_r )
    h_prime = a-h
    delta_z_0s1 = 270*( 1-np.tanh(0.28+1.2*np.sqrt(np.divide(h_prime,h))) )
    delta_z_0s2 = delta_z_0s1*( 1-np.tanh(float(1+np.divide( 0.48*np.power(np.divide(w_eff, h)-1,0.5), np.power(1+np.divide(h_prime, h), 2) ))) )

    if np.divide(w,h)>1.3:
        z_0 = z_0u-delta_z_0s1
    else:
        z_0 = z_0u-delta_z_0s2

    return z_0


def effectivePermittivityYa( h, w, e_r, f ):
    """
    Calculate frequency dependendt effective permittivity form Yamashita (dispersion).

    h   strip height over dielectric [M]
    w   strip width [M]
    e_r is the relative permittivity of the dielectric
    f   is the fignal frequency [Hz]
    - T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
    """
    e_eff = effectivePermittivityHJ( h, w, e_r )

    F = np.divide( 4*h*f*np.sqrt(e_eff-1), const.c ) * (0.5+np.power( 1+2*np.log10(1+np.divide(w,h)), 2 ))
    e_eff_freq = np.power(np.divide( np.sqrt(e_r)-np.sqrt(e_eff), 1+4*np.power(F,-1.5) ) + np.sqrt(e_eff), 2)
    return e_eff_freq


def microstripImpedanceYa( h, w, e_r, f ):
    """
    Calculate frequency dependendt Characteristic Impedance form Yamashita.

    h   strip height over dielectric [M]
    w   strip width [M]
    e_r is the relative permittivity of the dielectric
    f   is the fignal frequency [Hz]

    Accurate within 1% for 0.1 < f < 100 [GHz]
    - T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
    """
    e_eff_freq = effectivePermittivityYa(h, w, e_r, f)
    z_01 = Z01HJ(h, w, e_r)
    Z_0_freq = np.divide( z_01, np.sqrt(e_eff_freq) )
    return Z_0_freq


def microstripImpedanceKJ( h, w, e_r, f ):
    """
    Calculate frequency dependendt Characteristic Impedance form Kirschning and Jansen.

    h   strip height over dielectric [M]
    w   strip width
    e_r is the relative permittivity of the dielectric
    f   is the fignal frequency [Hz]

    Accurate within 0.6% for:
    f < 60 [GHz]
    1   <= e_r <= 20
    0.1 <= w/h <= 100
    0   <= h/lambda_0 <= 0.13
    - T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
    """

    e_eff = effectivePermittivityHJ( h, w, e_r)
    F = np.divide(f, 1e9)
    H = np.divide(h, 1e-2)
    P1 = 0.27488 + np.divide( 0.6315+0.525, np.power(1+0.157*F*H, 20))*np.divide(w,h) - 0.0065683*np.exp(-8.7513*np.divide(w,h))
    P2 = 0.33622*(1-np.exp(-0.03442*e_r))
    P3 = 0.0363*np.exp(-4.6*np.divide(w,h))*(1-np.exp(-np.power(np.divide(F*H,3.87),4.97)))
    P4 = 1+2.751*(1-np.exp(-np.power(np.divide(e_r,15.916),8)))

    def P(F):
        return P1*P2*np.power( (0.1844+P3*P4)*10*F*H, 1.5763 )

    e_eff_freq = e_r-np.divide(e_r-e_eff,1+P(F))
    z_01 = Z01HJ(h, w, e_r)
    Z_0_freq = np.divide( z_01, np.sqrt(e_eff_freq) )
    return Z_0_freq

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