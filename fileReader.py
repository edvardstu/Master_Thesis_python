import numpy as np


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

def loadFileNew(fileName):
    index, time_temp, x_temp, y_temp, theta_temp, vx_temp, vy_temp, D_r_temp, deformation_temp = np.loadtxt(fileName, unpack=True)
    n_particles = int(np.amax(index)) + 1
    n_steps = int(len(index) / n_particles)
    x = np.zeros([n_steps, n_particles])
    y = np.zeros([n_steps, n_particles])
    theta = np.zeros([n_steps, n_particles])
    vx = np.zeros([n_steps, n_particles])
    vy = np.zeros([n_steps, n_particles])
    time = np.zeros(n_steps)
    D_r = np.zeros(n_steps)
    deformation = np.zeros([n_steps, n_particles])

    for i in range(0, n_steps):
        for j in range(0, n_particles):
            #    time[i][j] = time_temp[i * n_particles + j]
            x[i][j] = x_temp[i * n_particles + j]
            y[i][j] = y_temp[i * n_particles + j]
            theta[i][j] = theta_temp[i * n_particles + j]
            vx[i][j] = vx_temp[i * n_particles + j]
            vy[i][j] = vy_temp[i * n_particles + j]
            deformation[i][j] = deformation_temp[i * n_particles + j]
        time[i] = time_temp[i * n_particles]
        D_r[i] = D_r_temp[i * n_particles]

    return time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation

def loadSeveralFiles(fileNameBase, numberOfFiles, startNumber):
    fileName = fileNameBase + str(startNumber) + ".txt"
    # time, x, y, theta, n_particles, n_steps, D_r = loadFile(fileName)

    time_f, x_f, y_f, theta_f, vx_f, vy_f, n_particles, n_steps, D_r_f = loadFile(fileName)


    for i in range(1+startNumber, numberOfFiles+startNumber):
        fileName = fileNameBase + str(i) + ".txt"
        time, x, y, theta, vx, vy, n_particles, n_steps, D_r = loadFile(fileName)

        time_f = np.concatenate((time_f, time), axis=0)
        x_f = np.concatenate((x_f, x), axis=0)
        y_f = np.concatenate((y_f, y), axis=0)
        theta_f = np.concatenate((theta_f, theta), axis=0)
        vx_f = np.concatenate((vx_f, vx), axis=0)
        vy_f = np.concatenate((vy_f, vy), axis=0)
        D_r_f = np.concatenate((D_r_f, D_r), axis=0)

    return time_f, x_f, y_f, theta_f, vx_f, vy_f, n_particles, n_steps, D_r_f