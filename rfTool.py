import numpy as np
import scipy.constants as const

class rfTool:
#    def __init__(self):
        
    # Calculate effective permittivity from Hammerstad-Jensen (simplified formula)
    def effectivePermittivityHJ(self, H, W, e_r):
        e_eff = (e_r+1)/2 + ((e_r-1)/2)*(1/np.sqrt(1+12*H/W))
        return e_eff

    # Calculate Z01 from Hammerstad-Jensen (simplified formula)
    def Z01HJ(self, H, W, e_r):
        e_eff = effectivePermittivityHJ( H, W, e_r)
        u = W/H
        F1 = 6+(2*np.pi-6)*np.exp(-(30.666/u)**0.7528)
        z_01 = 60*np.log( F1/u + np.sqrt(1+(2/u)**2) )
        z_0 = z_01/np.sqrt(e_eff)
        return z_0

    # Calculate Characteristic Impedance from Hammerstad-Jensen (simplified formula)
    def microstripImpedanceHJ(self, H, W, e_r):
        e_eff = effectivePermittivityHJ( H, W, e_r)
        z_01 = self.Z01HJ(H, W, e_r)
        z_0 = z_01/np.sqrt(e_eff)
        return z_0

    # Calculate frequency dependendt effective permittivity form Yamashita
    def effectivePermittivityYa(self, H, W, e_r, f):
        e_eff = effectivePermittivityHJ( H, W, e_r)

        F = ( (4*H*f*np.sqrt(e_eff-1))/(const.c) )*(0.5+(1+2*np.log10(1+(W/H)))**2)
        e_eff_freq = ( (np.sqrt(e_r)-np.sqrt(e_eff)) / (1+4*F**(-1.5)) + np.sqrt(e_eff) )**2
        return e_eff_freq

    # Calculate frequency dependendt Characteristic Impedance form Yamashita
    def microstripImpedanceYa(self, H, W, e_r, f):
        e_eff_freq = self.effectivePermittivityYa(H, W, e_r, f)
        z_01 = self.Z01HJ(H, W, e_r)
        Z_0_freq = np.divide( z_01 ,np.sqrt(e_eff_freq) )
        return Z_0_freq