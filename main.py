import numpy as np
from matplotlib import pyplot as plt
import fileReader
import scatterPlots
import derivedPlots



def main():
    barrier = scatterPlots.Barrier.PeriodicFunnel
    L=31.0
    H=31.0
    h=9.3
    R=30


    fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_funnel/funnelHigthSweep7.txt"
    #fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_tube/Dr0_2DensitySweep5.txt"
    time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
    n_frames = 200
    checkPosition(x, y, R, L, H, h, barrier)

    #derivedPlots.plotAvgV(time, vx, vy)
    #plt.show()

    scatterPlots.run_animation(x, y, theta, vx, vy, L ,H ,h, R, n_steps, n_particles, n_frames, barrier)

    #scatterPlots.plotState(x, y, vx, vy, 0)
    #plt.show()
    #scatterPlots.plotBoundary(barrier, L, H, 10, R)
    #scatterPlots.plotLimits(barrier, L, H, R)



    #time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
    #derivedPlots.plotAvgV(time, vx, vy)
    #plt.show()

    #fileNameBase = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_tube/"
    #derivedPlots.plotTimeAvgV(fileNameBase, 10)
    #derivedPlots.plotOrderParameterSqaure(fileNameBase, 10)
    #plt.show()

    #fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/JammingDr0_50.txt"
    #fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_tube/Dr0_5DensitySweep9.txt"
    #derivedPlots.calcFPSF(fileName)
    #fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_funnel/test0.txt"
   # derivedPlots.calcFPSF(fileName)

    #fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_tube/Dr0_2DensitySweep6.txt"
    #time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
    #n_frames = 200
    #scatterPlots.run_animation(x, y, theta, vx, vy, L ,H ,h, R, n_steps, n_particles, n_frames, barrier)

def checkPosition(x, y, R, L, H, h, barrier):
    factor = 1.1
    if barrier == scatterPlots.Barrier.Periodic:
        x_tf = np.abs(x) > factor * L
        y_tf = np.abs(y) > factor * H
    elif barrier == scatterPlots.Barrier.PeriodicTube:
        x_tf = np.abs(x) > factor * L
        y_tf = np.abs(y) > factor * H
    elif barrier == scatterPlots.Barrier.PeriodicFunnel:
        x_tf = np.abs(x) > factor * L
        y_tf = np.abs(y) > ((np.abs(x)*(H-h)/L) + h/2)
    elif barrier == scatterPlots.Barrier.Circular:
        x_tf = np.sqrt(x**2 + y**2) > factor * R
        y_tf = x_tf

    if (x_tf.any()):
        print("Particles are out of the boundary in the x-direction")
    if (y_tf.any()):
        print("Particles are out of the boundary in the y-direction")

main()


def calcFPSF():
    fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/fpsf0Susc.txt"
    time, Q_t = np.loadtxt(fileName, unpack=True)
    print(Q_t.var())

#calcFPSF()

