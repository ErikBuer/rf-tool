
import numpy as np
import scipy.optimize as optimize

def chamberImpedance( x ):
    """
    Dimension calculations for an open TEM cell.
    S. M. Satav et al., Do-it-Yourself Fabrication of an Open TEM Cell for EMC Pre-compliance, Indian Institute of Technology-Bombay, 2008
    Taking in a parameter vector for the different dimensions.
    
    Chamber below, with ceptum in the center.
     _______
    /_______\
    \_______/
    """

    
    d = x[0]    # Height from center conductor to top [m]. Entire chamber is 2d high.
    W = x[1]    # Width of the center conductor (septum) [m].
    L = x[2]    # Length of the test area [m].

    # Assuming air in the chamber
    e_r = 1 # Permittivity in the chaimber (Air)
    targetImpedance = 50 #Ohm
    t = 1.6e-3  #the thickness of the center conductor (ceptum).
    C_f = L*0.053e-10 # Fringing capacitance per unit length [F/m] (0.053 pF/cm). Unshure what is ment by the unit length. 
    #Assuming length of the straight (non-tapered) portionn of the chamber.

    Z0=np.divide(94.15,np.sqrt(e_r)*( np.divide(W,2*d*(1-np.divide(t,2*d))) + np.divide(C_f, 0.0885*e_r) ))

    cost = np.power(np.abs(targetImpedance-Z0), 2)
    return cost


def chamberDimensions( minHeight=10e-2, minWidth=30e-2, minLength=30e-2 ):
    """
    Optimization routine for solving the physical dimensions for the chamber which satisifies a 50 ohm impedance.
    minHeight is the height of the intended test object.
    minWidth is the x and y dimention of the intended test area.
    """
    x0 = np.array([minHeight, minWidth, minLength])

    """
    d = x[0]    # Height from center conductor to top [m]. Entire chamber is 2d high.
    W = x[1]    # Width of the center conductor (septum) [m].
    L = x[2]    # Length of the test area [m].
    """

    bnds = ((minHeight, None), (minWidth, None), (minLength, None))
    res = optimize.minimize( chamberImpedance, x0, bounds=bnds, method='L-BFGS-B' )
    x = res.x   #Optimized parameters

    print("Test area height, d =", x[0])
    print("Septum Width (test area width), W =", x[1])
    print("Test area Length, L =", x[2])
