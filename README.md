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

Z_static = rf.microstripImpedanceHJ(0.1, 0.1, 4.2)
Z_100M = rf.microstripImpedanceYa(0.1, 0.1, 4.2, 100e6)
```
