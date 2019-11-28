# rf-tool
RF electronics calculator

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
# Static impedance
Z_static = rf.microstripImpedanceHJ( h, w, e_r )

# Frequency dependent impedance calculation (Yamashita dispersion)
Z_100M = rf.microstripImpedanceYa( h, w, e_r, f)

# Frequency dependent impedance calculation (Kirschning and Jansen dispersion)
Z_100M = rf.microstripImpedanceKJ( h, w, e_r, f):

# Albersheim equation for required SNR
SNRdB = rf.Albersheim( Pfa, Pd, N )
```
