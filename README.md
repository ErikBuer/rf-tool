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
import rf-tool as rf
# Static impedance
Z_static = rf.microstripImpedanceHJ( H, W, e_r )

# Frequency dependent impedance calculation
Z_100M = rf.microstripImpedanceYa( H, W, e_r, f)

# Albersheim equation for required single pulse SNR
SNRdB = rf.Albersheim( Pfa, Pd, N )
```
