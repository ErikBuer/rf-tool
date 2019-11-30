# rf-tool
RF electronics calculator.
Developed for python 3.

Calculate static or frequency dependent characteristic impedance for microstip lines.
Based on the empirical effective permittivity and impedance formulas of Hammerstad-Jensen, and the empirical dispersion formula of yamashita.

## Available on PYPI
https://pypi.org/project/rf-tool/
```
pip install rf-tool
```

## Basic Usage:

```
import rftool as rf
```
Use the help() function for description of inputs and valid ranges of the functions.

### PCB Tools
```
# Quasi static impedance, Hammerstad and Jensen's method.
Z_static = rf.microstripImpedanceHJ( h, w, e_r )

# Calcultate the effective width, accounting for the microstrip height, Hammerstad and Jensen's method.
w_eff = rf.effectiveStripWidthHJ( h, w, t, e_r )

# Quasi static impedance of microstrip in metallic enclosure.
Z_static = rf.shieldedMicrostripImpedanceHJ( h, w, t, a, b, e_r )

# Frequency dependent impedance calculation (Yamashita dispersion)
Z_100M = rf.microstripImpedanceYa( h, w, e_r, f)

# Frequency dependent impedance calculation (Kirschning and Jansen dispersion)
Z_100M = rf.microstripImpedanceKJ( h, w, e_r, f)

```

### Radar Tools
```
# Albersheim's equation for required SNR with incoherent integration
SNRdB = rf.Albersheim( Pfa, Pd, N )

# Shnidman's equation for required SNR with incoherent integration, swerling 0-5
SNRdB = Shnidman( Pfa, Pd, N, SW )
```

### Utility
```
# Conversion between reflection coef. and VSWR
VSWR = rf.Gamma2VSWR( Gamma )

# Linear to log conversion 
dB = rf.mag2db( mag )
mag = rf.db2mag( dB )

dB = rf.pow2db( power )
mag = rf.db2Pow( dB )
```

