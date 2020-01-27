import numpy as np
from matplotlib import pyplot as plt


def loadFile(fileName):
    index, time_temp, x_temp, y_temp, theta_temp, vx_temp, vy_temp, D_r_temp = np.loadtxt(fileName, unpack=True)
    n_particles = int(np.amax(index)) + 1
    n_steps = int(len(index) / n_particles)
    x = np.zeros([n_steps, n_particles])
    y = np.zeros([n_steps, n_particles])
    theta = np.zeros([n_steps, n_particles])
    vx = np.zeros([n_steps, n_particles])
    vy = np.zeros([n_steps, n_particles])
    time = np.zeros(n_steps)
    D_r = np.zeros(n_steps)

    for i in range(0, n_steps):
        for j in range(0, n_particles):
            #    time[i][j] = time_temp[i * n_particles + j]
            x[i][j] = x_temp[i * n_particles + j]
            y[i][j] = y_temp[i * n_particles + j]
            theta[i][j] = theta_temp[i * n_particles + j]
            vx[i][j] = vx_temp[i * n_particles + j]
            vy[i][j] = vy_temp[i * n_particles + j]
        time[i] = time_temp[i * n_particles]
        D_r[i] = D_r_temp[i * n_particles]

    return time, x, y, theta, vx, vy, n_particles, n_steps, D_r

'''
plt.figure()

fileName = "Results/Integrators/EMSimple0.txt"
time, x, y, theta, n_particles, n_steps, D_r = loadFile(fileName)
plt.plot(time, x, label='EM')

fileName = "Results/Integrators/ABSimple0.txt"
time, x, y, theta, n_particles, n_steps, D_r = loadFile(fileName)
plt.plot(time, x, label='AB')


plt.legend()
plt.show()

'''