import numpy as np
import h5py


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

def loadSimulationParameters(fileName):
    #fileName = fileName.replace(".txt", "SimulationParameters.txt")
    f = open(fileName, 'r')

    data = []
    for line in f.readlines():
        data.append(line.replace('\n', '').split(' '))

    f.close()

    R = float(data[1][3])
    L = float(data[2][3])
    H = float(data[3][3])
    h = float(data[4][3])
    r_a = float(data[5][3])
    n_particles = int(data[6][3])
    n_fixed_particles = int(data[7][4])
    u_0 = float(data[8][2])
    D_r = float(data[9][2])

    n_steps = int(data[12][3])
    dt = float(data[13][2])

    return R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt

#print(loadSimulationParameters("/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/phaseDiagram/testHDF50.txt"))

def loadFileHDF5(fileNameBase):
    fileNameSimPar = fileNameBase + "SimulationParameters.txt"

    R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt = loadSimulationParameters(fileNameSimPar)

    fileNameH5 = fileNameBase + ".h5"
    file = h5py.File(fileNameH5, "r")
    dataset = np.array(file["dset"])
    n_data_points = dataset.shape[0]
    n_written_steps = int(n_data_points/n_particles)
    write_interval = int(n_steps/(n_written_steps-1))
    time = dataset[0:n_data_points:n_particles, 1]
    x = np.reshape(dataset[:,2], (n_written_steps, n_particles))
    y = np.reshape(dataset[:,3], (n_written_steps, n_particles))
    theta = np.reshape(dataset[:,4], (n_written_steps, n_particles))
    vx = np.reshape(dataset[:,5], (n_written_steps, n_particles))
    vy = np.reshape(dataset[:,6], (n_written_steps, n_particles))

    return time, x, y, theta, vx, vy, n_particles, n_steps, D_r, u_0, dt, write_interval, n_written_steps


def loadFPSFandKurtosis(fileNameBase):
    fileName = fileNameBase + "FpsfAndKurtosis.txt"
    tau, Q_t_time_avg, chi_4, kurtosis = np.loadtxt(fileName, unpack=True)
    return tau, Q_t_time_avg, chi_4, kurtosis

