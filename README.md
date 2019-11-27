# rfTool
RF electronics calculator

Calculate static or frequency dependent characteristic impedance for microstip lines.
Based on the empirical formulas of Hammerstad-Jensen and yamashita.

## Basic Usage:
```
import rfTool as rfT

Z_static = rfT.microstripImpedanceHJ(0.1, 0.1, 4.2)
Z_100M = rfT.microstripImpedanceYa(0.1, 0.1, 4.2, 100e6)
```
