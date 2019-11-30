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
Use the help() function for description of inputs and valid range.

```
import rftool as rf
# Static impedance
Z_static = rf.microstripImpedanceHJ( h, w, e_r )

# Frequency dependent impedance calculation (Yamashita dispersion)
Z_100M = rf.microstripImpedanceYa( h, w, e_r, f)

# Frequency dependent impedance calculation (Kirschning and Jansen dispersion)
Z_100M = rf.microstripImpedanceKJ( h, w, e_r, f):

# Albersheim's equation for required SNR with incoherent integration
SNRdB = rf.Albersheim( Pfa, Pd, N )

# Shnidman's equation for required SNR with incoherent integration, swerling 0-5
SNRdB = Shnidman( Pfa, Pd, N, SW ):
```
