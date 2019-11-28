import numpy as np
import scipy.constants as const

        
# Calculate effective permittivity from Hammerstad-Jensen (simplified formula)
"""
- T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
"""
def effectivePermittivityHJ(h, w, e_r):
    e_eff = np.divide((e_r+1),2) + np.divide((e_r-1),2)*np.divide(1,np.sqrt(1+(12*np.divide(h,w))))
    return e_eff

# Calculate Z01 from Hammerstad-Jensen (simplified formula)
"""
h   strip height over dielectric
w   strip width
e_r is the relative permittivity of the dielectric
- T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
"""
def Z01HJ( h, w, e_r):
    e_eff = effectivePermittivityHJ( h, w, e_r)
    u = np.divide(w,h)
    F1 = 6+(2*np.pi-6)*np.exp(-np.power(np.divide(30.666,u), 0.7528))
    z_01 = 60*np.log( np.divide(F1, u) + np.sqrt(1+np.power(np.divide(2,u),2)) )
    return z_01

# Calculate Characteristic Impedance from Hammerstad-Jensen (simplified formula)
"""
h   strip height over dielectric
w   strip width
e_r is the relative permittivity of the dielectric
- T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
"""
def microstripImpedanceHJ( h, w, e_r):
    e_eff = effectivePermittivityHJ( h, w, e_r)
    z_01 = Z01HJ(h, w, e_r)
    z_0 = np.divide(z_01,np.sqrt(e_eff))
    return z_0

# Calculate frequency dependendt effective permittivity form Yamashita (dispersion)
"""
h   strip height over dielectric [M]
w   strip width [M]
e_r is the relative permittivity of the dielectric
f   is the fignal frequency [Hz]
- T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
"""
def effectivePermittivityYa( h, w, e_r, f):
    e_eff = effectivePermittivityHJ( h, w, e_r)

    F = np.divide( (4*h*f*np.sqrt(e_eff-1)), (const.c) )*(0.5+np.power(1+2*np.log10(1+np.divide(w,h)),2))
    e_eff_freq = np.power(np.divide( (np.sqrt(e_r)-np.sqrt(e_eff)), 1+4*np.power(F,-1.5)) + np.sqrt(e_eff), 2)
    return e_eff_freq

# Calculate frequency dependendt Characteristic Impedance form Yamashita
"""
h   strip height over dielectric [M]
w   strip width [M]
e_r is the relative permittivity of the dielectric
f   is the fignal frequency [Hz]
Accurate within 1% for 0.1 < f < 100 [GHz]
- T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
"""
def microstripImpedanceYa( h, w, e_r, f):
    e_eff_freq = effectivePermittivityYa(h, w, e_r, f)
    z_01 = Z01HJ(h, w, e_r)
    Z_0_freq = np.divide( z_01, np.sqrt(e_eff_freq) )
    return Z_0_freq

# Calculate frequency dependendt Characteristic Impedance form Kirschning and Jansen
"""
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
def microstripImpedanceKJ( h, w, e_r, f):
    e_eff = effectivePermittivityHJ( h, w, e_r)
    F = np.divide(h, 1e9)
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

# Calculate required SNR for non-coherent integration over N pulses, by use of Albersheims equation
"""
Accurate within 0.2 dB for:
10^-7   <  Pfa  < 10^-3
0.1     <  Pd   < 0.9
1       <= N    < 8096
- M. A. Richards and J. A. Scheer and W. A. Holm, Principles of Modern Radar, SciTech Publishing, 2010 
"""
def Albersheim( Pfa, Pd, N ):
    A = np.log(np.divide(0.062, Pfa))
    B = np.log(np.divide(Pd,1-Pd))
    SNRdB = -5*np.log10(N)+(6.2+np.divide(4.54, np.sqrt(N+0.44)))*np.log10(A+(0.12*A*B)+(0.7*B))
    return SNRdB
