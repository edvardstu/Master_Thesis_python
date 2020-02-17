import numpy as np
from matplotlib import pyplot as plt
import fileReader

def plotRelativeErrorEM(fileNameBase, isAB):
    counter = 0
    sigma_pp = 1.5
    r_cutoff = np.sqrt(2)*sigma_pp #2*sigma_pp*sigma_pp

    solverType = "EM"
    if isAB:
        solverType = "AB"

    fileName = fileNameBase + solverType + str(counter) + ".txt"
    time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)

    for i in range(0, 3):
        counter+=1
        fileName = fileNameBase + solverType + str(counter) + ".txt"
        time_next, x_next, y_next, theta_next, vx_next, vy_next, n_particles_next, n_steps_next, D_r_next, deformation_next = fileReader.loadFileNew(fileName)

        delta_r_vec = np.sqrt((x_next-x)**2 + (y_next-y)**2)
        t = np.linspace(0, 1, num=1001)

        delta_r = np.mean(delta_r_vec/r_cutoff, axis=1)
        label = r'%s $\Delta t*=10^{%d}$, $\Delta t=10^{%d}$' % (solverType, -counter-3, -counter-2)
        plt.figure(0)
        plt.semilogy(t, delta_r, label=label)

        '''
        if (i!=0):
            step = 40/400
            der_delta_r = (delta_r[1:401]-delta_r[0:400])/step
            plt.figure(1)
            plt.semilogy(time[1:401], der_delta_r, label=label)
        '''

        x = x_next
        y = y_next


    plt.figure(0)
    plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\langle |r*-r|\rangle/r_c$ ')
    #picName = fileNameBase + solverType + ".png"
    #plt.savefig(picName, dpi=150)

    '''
    plt.figure(1)
    plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$(d/dt)\langle |r*-r|\rangle/r_c$ ')
    picName = fileNameBaseEM + "Derivative.png"
    plt.savefig(picName, dpi=150)
    '''
    #plt.show()

fileNameBase = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/benchmark/benchmark"
plotRelativeErrorEM(fileNameBase, False)
plotRelativeErrorEM(fileNameBase, True)
plt.savefig("plots/benchmark/ABvsEM.png", dpi=200)
plt.show()
