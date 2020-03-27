import numpy as np
from matplotlib import pyplot as plt
import fileReader
import scatterPlots
import derivedPlots



def main():
    barrier = scatterPlots.Barrier.Periodic
    L=25
    H=25
    h=0
    R=30

    '''
    fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/Dr0_2DensitySweep0.txt"
    time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
    n_frames = 200

    x_tf = np.abs(x)>2*L
    if (x_tf.any()):
        print("Particles are out of the boundary in the x-direction")
    y_tf = np.abs(y) > 2 * H
    if (y_tf.any()):
        print("Particles are out of the boundary in the y-direction")

    #derivedPlots.plotAvgV(time, vx, vy)

    scatterPlots.run_animation(x, y, theta, vx, vy, L ,H ,h, R, n_steps, n_particles, n_frames, barrier)

    #scatterPlots.plotState(x, y, vx, vy, 150)
    #scatterPlots.plotBoundary(barrier, L, H, 10, R)
    #scatterPlots.plotLimits(barrier, L, H, R)
    '''

    #plt.figure()
    #fileNameBase = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/"
    #derivedPlots.plotTimeAvgV(fileNameBase, 10)
    #derivedPlots.plotOrderParameterSqaure(fileNameBase, 10)
    #plt.show()

    fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/JammingDr0_50.txt"
    #fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/Dr0_5DensitySweep9.txt"
    derivedPlots.calcFPSF(fileName)
    #plt.figure()
    #fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/Dr0_5DensitySweep9.txt"
    #derivedPlots.calcFPSF(fileName)
    #plt.show()

main()

def calcFPSF():
    fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/fpsf0Susc.txt"
    time, Q_t = np.loadtxt(fileName, unpack=True)
    print(Q_t.var())

#calcFPSF()

