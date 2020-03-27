import numpy as np
from matplotlib import pyplot as plt

def forceHarmonic(r, r_c, r_a, v_0):
    return -v_0*(r-r_c)/(r_c-r_a)

def forceLennardJonesRep(r):
    return 12.0/r**13

def forceLennardJonesShifted(r, sigma, y_0):
    return 4*(12*sigma**12/r**13-6*sigma**6/r**7)+y_0

def forceLennardJonesReduced(r, sigma):
    return 12*(2*sigma**2/r**3-1/r)

def plotForces():
    r_a=1
    v_0=1

    r_max = 2.0
    r = np.linspace(0.01, r_max, 1000)

    r_c=1.1
    plt.plot(r, forceHarmonic(r, r_c, r_a, v_0), label= 'Harmonic')

    plt.plot(r, forceLennardJonesRep(r), label='LennardJonesRep')

    y_0 = 504/169*(7/13)**(1/6)
    sigma = 2**(-1/6)
    plt.plot(r, forceLennardJonesShifted(r, sigma, y_0), label='LennardJonesShfited')

    sigma = 2 ** (-1 / 2)
    plt.plot(r, forceLennardJonesReduced(r, sigma), label='LennardJonesReduced')
    #plt.grid()
    plt.grid(True, which='both', linestyle='--')

    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.legend()
    plt.ylim(-1.0, 10.0)
    plt.show()


plotForces()