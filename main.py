import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import fileReader
import scatterPlots
import derivedPlots

R = 17
fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/fixed_boundary/testForceChains0.txt"
#fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/fixed_boundary/test0.txt"
#fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/fixed_boundary/testAB0.txt"
#fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/fixed_boundary/testABFineSteps0.txt"

time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
#time, x, y, theta, vx, vy, n_particles, n_steps, D_r = fileReader.loadFile(fileName)

frame_number = 500
#scatterPlots.plotDeformationMap(x, y, deformation, frame_number, R)
scatterPlots.plotVelocityMap(x, y, vx, vy, frame_number, R)

n_frames = 200
#scatterPlots.run_animation(x, y, theta, R, n_steps, n_particles, n_frames)

#derivedPlots.plotOrderParameter2(time, x, y, theta, vx, vy, D_r)

plt.show()


