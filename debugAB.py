import numpy as np
from matplotlib import pyplot as plt

def loadFile(fileName):
    index, time_temp, x_temp, y_temp, theta_temp, vx_temp, vy_temp = np.loadtxt(fileName, unpack=True)
    n_particles = int(np.amax(index)) + 1
    n_steps = int(len(index) / n_particles)
    x = np.zeros([n_steps, n_particles])
    y = np.zeros([n_steps, n_particles])
    theta = np.zeros([n_steps, n_particles])
    vx = np.zeros([n_steps, n_particles])
    vy = np.zeros([n_steps, n_particles])
    time = np.zeros(n_steps)


    for i in range(0, n_steps):
        for j in range(0, n_particles):
            #    time[i][j] = time_temp[i * n_particles + j]
            x[i][j] = x_temp[i * n_particles + j]
            y[i][j] = y_temp[i * n_particles + j]
            theta[i][j] = theta_temp[i * n_particles + j]
            vx[i][j] = vx_temp[i * n_particles + j]
            vy[i][j] = vy_temp[i * n_particles + j]
        time[i] = time_temp[i * n_particles]

    return time, x, y, theta, vx, vy, n_particles, n_steps


def analytical(t):
    #sigma_pp = 1.5
    #d = np.sqrt(2*sigma_pp**2)
    d = 1.0
    v_0 = 1
    k = 1.0
    z_0 = 1.0

    return z_0*np.exp(-2*k*t) + (d-v_0/k)*(1-np.exp(-2*k*t))




fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/AB_test/testEM.txt"
time, x, y, theta, vx, vy, n_particles, n_steps = loadFile(fileName)
#plt.plot(time, x[:, 1]-x[:, 0], "-x", label="EM")
#plt.plot(x, y, "-x", label="EM")

d_r = np.sqrt((x[:, 0]-x[:, 1])**2+(y[:, 0]-y[:, 1])**2)
plt.plot(time, d_r, "-x", label="EM")

fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/AB_test/testAB.txt"
time, x, y, theta, vx, vy, n_particles, n_steps = loadFile(fileName)
#plt.plot(time, x[:, 1]-x[:, 0], "-x", label="AB")
#plt.plot(x, y, "-x", label="AB")

d_r = np.sqrt((x[:, 0]-x[:, 1])**2+(y[:, 0]-y[:, 1])**2)
plt.plot(time, d_r, "-x", label="AB")

#n_points = 31
#time = np.linspace(0.0, 3.0, num=n_points)
#x_analytic = analytical(time)
#plt.plot(time, x_analytic, "-x", label="An")

plt.legend()
plt.show()