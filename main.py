import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import fileReader
import scatterPlots
import derivedPlots
import enum

R = 30#17
#fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/fixed_boundary/testForceChains0.txt"
#fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/fixed_boundary/test0.txt"
#fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/fixed_boundary/testAB0.txt"
#fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/fixed_boundary/testABFineSteps0.txt"

'''
fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/benchmark/benchmarkEM0.txt"
time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
scatterPlots.plotState(x, y, vx, vy, 1000, R)


fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/benchmark/benchmarkAB0.txt"
time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
scatterPlots.plotState(x, y, vx, vy, 1000, R)
'''
#time, x, y, theta, vx, vy, n_particles, n_steps, D_r = fileReader.loadFile(fileName)

#frame_number = 500

#fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/fpsf0.txt"
#time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
#n_frames =100
#R=35
#L=35
#H=35

#plt.figure()
#plt.semilogy(time, np.amax(vx, axis=1))
#plt.semilogy(time, np.amax(vy, axis=1))

#scatterPlots.run_animation_tube(x, y, theta, vx, vy, L, H, n_steps, n_particles, n_frames)
#scatterPlots.plotStateSqaure(x, y, vx, vy, 10, L, H)

class Barrier(enum.Enum):
    Circular = 1
    Periodic = 2
    PeriodicTube = 3
    PeriodicFunnel = 4

def main():
    barrier = Barrier.PeriodicTube
    L=45
    H=35
    R=35

    boundary = np.array([None, None])
    boundary_periodic = np.array([None, None])
    if barrier==Barrier.Circular:
        n_points = 100
        d_theta = 2*np.pi/100
        theta1 = np.linspace(0, 2*np.pi-d_theta, n_points)
        theta2 = np.linspace(d_theta, 2 * np.pi, n_points)
        boundary = np.array([R*np.cos(theta1), R*np.cos(theta2), R*np.sin(theta1), R*np.sin(theta2)])
    elif barrier==Barrier.Periodic:
        boundary_periodic = np.array([[-L/2, -L/2, L/2, -L/2],
                            [L/2, L/2, L/2, -L/2],
                            [H/2, -H/2, -H/2, -H/2],
                            [H/2, -H/2, H/2, H/2]])
    elif barrier==Barrier.PeriodicTube:
        boundary= np.array([[-L/2, -L/2],
                            [L/2, L/2],
                            [H/2, -H/2],
                            [H/2, -H/2]])
        boundary_periodic= np.array([[L/2, -L/2],
                                    [L/2, -L/2],
                                    [H/2, H/2],
                                    [-H/2, -H/2]])


    if (boundary.any()!=None):
        n_boundary = len(boundary[0])
        for i in range(0, n_boundary):
            plt.plot([boundary[0,i], boundary[1,i]],[boundary[2,i], boundary[3,i]], color='black')

    if (boundary_periodic.any()!=None):
        n_boundary_periodic = len(boundary_periodic[0])
        for i in range(0, n_boundary_periodic):
            plt.plot([boundary_periodic[0,i], boundary_periodic[1,i]],[boundary_periodic[2,i], boundary_periodic[3,i]], color='black', linestyle='dashed')
    plt.show()

main()
#plt.show()


