import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import fileReader
import scatterPlots
import derivedPlots

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

fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/ML/testData0.txt"
time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
n_frames = 100
R=4
scatterPlots.run_animation(x, y, theta, R, n_steps, n_particles, n_frames)


plt.show()


