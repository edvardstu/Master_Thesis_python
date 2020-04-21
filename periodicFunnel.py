from matplotlib import pyplot as plt
import numpy as np
import fileReader

def plotVelocityVsFunnelHigth(fileNameBase):
    L = 25
    H = 25
    numPoints = 8
    v_avg_vec = np.zeros(numPoints)
    h_vec = np.zeros(numPoints)
    for i in range(0,numPoints):
        fileName = fileNameBase + str(i) + ".txt"
        time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
        start = int(20000 / 200)
        vx = vx[start:n_steps, :]
        vy = vy[start:n_steps, :]
        #v_avg_vec[i]=np.mean(np.sqrt(vx**2+vy**2))
        v_avg_vec[i] = np.mean(np.abs(vx))
        f_h = (20-2*i)/20
        #h_vec[i] = np.sqrt(L*H * 2 / (1 + f_h) * H / L)
        h_vec[i]=f_h

    plt.plot(h_vec, v_avg_vec)
    plt.xlabel(r"$H_{f}/H$")
    plt.ylabel(r"$\langle |v_{x}| \rangle_{r,t}$")
    plt.show()


#fileNameBase = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_funnel/funnelHigthSweep"
#plotVelocityVsFunnelHigth(fileNameBase)

def plotDensityFlow(fileName, label):
    time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
    f_time_interval=4
    x_crossed = np.zeros(n_steps-f_time_interval)
    for i in range(0, n_steps-f_time_interval):
        x_temp = ((x[i]>0) & (x[i+f_time_interval]<0)) + ((x[i]<0) & (x[i+f_time_interval]>0))
        x_crossed[i] = np.sum(x_temp)

    plt.plot(time[0:n_steps-f_time_interval], x_crossed, label=label)
    #plt.show()


fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_funnel/funnelHigthSweep0.txt"
plotDensityFlow(fileName, r"$H_{f}/H=1.0$")
fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_funnel/funnelHigthSweep4.txt"
plotDensityFlow(fileName, r"$H_{f}/H=0.6$")
fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_funnel/funnelHigthSweep7.txt"
plotDensityFlow(fileName, r"$H_{f}/H=0.3$")
plt.legend()
plt.show()