import numpy as np
import scipy.constants as const

        
# Calculate effective permittivity from Hammerstad-Jensen (simplified formula)
"""
- T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
"""
def effectivePermittivityHJ(H, W, e_r):
    e_eff = np.divide((e_r+1),2) + np.divide((e_r-1),2)*np.divide(1,np.sqrt(1+(12*np.divide(H,W))))
    return e_eff

# Calculate Z01 from Hammerstad-Jensen (simplified formula)
"""
H   strip height over dielectric
W   strip width
e_r is the relative permittivity of the dielectric
- T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
"""
def Z01HJ( H, W, e_r):
    e_eff = effectivePermittivityHJ( H, W, e_r)
    u = np.divide(W,H)
    F1 = 6+(2*np.pi-6)*np.exp(-np.power(np.divide(30.666,u), 0.7528))
    z_01 = 60*np.log( np.divide(F1, u) + np.sqrt(1+np.power(np.divide(2,u),2)) )
    return z_01

# Calculate Characteristic Impedance from Hammerstad-Jensen (simplified formula)
"""
H   strip height over dielectric
W   strip width
e_r is the relative permittivity of the dielectric
- T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
"""
def microstripImpedanceHJ( H, W, e_r):
    e_eff = effectivePermittivityHJ( H, W, e_r)
    z_01 = Z01HJ(H, W, e_r)
    z_0 = np.divide(z_01,np.sqrt(e_eff))
    return z_0

# Calculate frequency dependendt effective permittivity form Yamashita (dispersion)
"""
H   strip height over dielectric
W   strip width
e_r is the relative permittivity of the dielectric
f   is the fignal frequency [Hz]
- T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
"""
def effectivePermittivityYa( H, W, e_r, f):
    e_eff = effectivePermittivityHJ( H, W, e_r)

    F = np.divide( (4*H*f*np.sqrt(e_eff-1)), (const.c) )*(0.5+(1+2*np.log10(1+np.divide(W,H)))**2)
    e_eff_freq = np.divide( (np.sqrt(e_r)-np.sqrt(e_eff)), (1+4*F**(-1.5)) + np.sqrt(e_eff) )**2
    return e_eff_freq

# Calculate frequency dependendt Characteristic Impedance form Yamashita
"""
H   strip height over dielectric
W   strip width
e_r is the relative permittivity of the dielectric
f   is the fignal frequency [Hz]
Accurate within 1% for 0.1 < f < 100 [GHz]
- T. C. Edwards and M. B. Steer, Foundations for microstrip circuit design, fourth edition, Wiley, 2016
"""
def microstripImpedanceYa( H, W, e_r, f):
    e_eff_freq = effectivePermittivityYa(H, W, e_r, f)
    z_01 = Z01HJ(H, W, e_r)
    Z_0_freq = np.divide( z_01 ,np.sqrt(e_eff_freq) )
    return Z_0_freq


# Calculate required single pulse SNR for non-coherent integration over N pulses, by use of Albersheims equation
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
