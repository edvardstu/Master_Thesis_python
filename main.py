import numpy as np
from matplotlib import pyplot as plt
import fileReader
import scatterPlots
import derivedPlots



def main():

    barrier = scatterPlots.Barrier.Periodic
    L=25.0
    H=25.0
    h=9.3
    R=30


    #fileNameBase = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/phaseDiagram/gamma_pp_0_0/testHDF5"
    #fileNameBase = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/phaseDiagram/gamma_pp_0_0/00/sweepPropulsion"
    #fileNameBase = "/media/edvardst/My Book/NTNU/Programming/Master_Thesis/Periodic_2D/phaseDiagram/gamma_pp_0_0/highDensity/sweepPropulsion"
    fileNameBase = "/media/edvardst/My Book/NTNU/Programming/Master_Thesis/Periodic_2D/phaseDiagram/gamma_pp_0_0/test/sweepPropulsion"

    #derivedPlots.calcFPSFandKurtisis(fileNameBase, 1, 0)

    #derivedPlots.plotFPSFandKurtisis(fileNameBase, 8)

    #derivedPlots.plotOrderParameterParallell(fileNameBase, 8)


    #for i in range(12):
    #    R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt = fileReader.loadSimulationParameters(fileNameBase + str(i) + "SimulationParameters.txt")
    #    print("{} {}".format(i, u_0))

    #folderName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/phaseDiagram/gamma_pp_0_0/"
    #fileName = "sweepPropulsion"
    #derivedPlots.plotFPSFandKurtisisSeveral(folderName, fileName, 12)

    n_frames = 100#20000

    derivedPlots.findFileName(fileNameBase, 0.4)
    fileName = "/media/edvardst/My Book/NTNU/Programming/Master_Thesis/Periodic_2D/phaseDiagram/gamma_pp_0_0/test/sweepPropulsion0"

    derivedPlots.plotAvgEnergy(fileName)
    derivedPlots.plotHistogram(fileName)

    plt.show()
    #checkPosition(x, y, R, L, H, h, barrier)
    #scatterPlots.run_animation(x, y, theta, vx, vy, L ,H ,h, R, n_written_steps, n_particles, n_frames, barrier)





    #scatterPlots.plotState(x, y, vx, vy, 0)
    #plt.show()
    #scatterPlots.plotBoundary(barrier, L, H, 10, R)
    #scatterPlots.plotLimits(barrier, L, H, R)


    #derivedPlots.plotAvgV(time, vx, vy)
    #derivedPlots.plotAvgEnergy(time, vx, vy, 0.2)
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
    factor = 1.0
    if barrier == scatterPlots.Barrier.Periodic:
        x_tf = np.abs(x) > factor * L/2
        y_tf = np.abs(y) > factor * H/2
    elif barrier == scatterPlots.Barrier.PeriodicTube:
        x_tf = np.abs(x) > factor * L/2
        y_tf = np.abs(y) > factor * H/2
    elif barrier == scatterPlots.Barrier.PeriodicFunnel:
        x_tf = np.abs(x) > factor * L/2
        #Not right I think
        y_tf = np.abs(y) > ((np.abs(x)*(H-h)/L) + h/2)
    elif barrier == scatterPlots.Barrier.Circular:
        x_tf = np.sqrt(x**2 + y**2) > factor * R
        y_tf = x_tf

    if (x_tf.any()):
        print("Particles are out of the boundary in the x-direction")
    if (y_tf.any()):
        print("Particles are out of the boundary in the y-direction")

main()



