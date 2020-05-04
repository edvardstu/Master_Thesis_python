import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import fileReader
import math
from datetime import datetime


def colourSet(n, cmap='gist_rainbow'):
    colourSet = []
    cmap_large = cm.get_cmap(cmap)
    palette = np.linspace(0.05, 0.95, n)
    for colourID in range(n):
        colourSet.append(cmap_large(palette[colourID]))
    return colourSet


def plotAvgV(time, vx, vy):
    v = np.sqrt(vx**2 + vy**2)
    v = np.mean(v, axis=1)
    plt.figure()
    plt.plot(time, v)
    #plt.ylim(0, 6)
    #plt.semilogy(time, v)
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'$\langle v \rangle_{r}$')

def plotAvgEnergy(time, vx, vy, v_0):
    E_0 = v_0**2
    start = int(20000 / 200)
    n_steps = len(time)
    vx = vx[start:n_steps, :]
    vy = vy[start:n_steps, :]
    E = np.mean(vx**2+vy**2, axis=1)
    plt.figure()
    plt.plot(time[start:n_steps], E/E_0)
    #plt.ylim(0, 6)
    #plt.semilogy(time, v)
    plt.xlabel(r'Time $t$')
    #plt.ylabel(r'$\langle E \rangle_{r}\cdot(2/m)$')
    plt.ylabel(r'$\langle E \rangle_{r}/E_{0}$')
    #E_0 free particle energy = 1/2 m v_0**2


def calcFPSFandKurtisis(fileNameBase):
    start_t = datetime.now()
    start = int(20000 / 20)
    max_steps_tau = 300
    steps_lin = np.floor(max_steps_tau/2)
    steps_log = np.ceil(max_steps_tau/2)
    tau_temp_1 = np.linspace(1, steps_lin, steps_lin, dtype=int)
    tau_temp_2 = np.geomspace(steps_lin+1, 15000, steps_log, dtype=int)
    tau_integers = np.concatenate((tau_temp_1, tau_temp_2))
    #U_0_array = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

    delta = 0.3 #0.7

    n_files = 8
    start_file = 0
    h_p_array = np.zeros(n_files)
    u_0_array = np.zeros(n_files)
    for i in range(0, n_files):
        fileName = fileNameBase + str(i+start_file)# + ".h5"
        #time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
        time, x, y, theta, vx, vy, n_particles, n_steps, D_r, u_0, dt, write_interval, n_written_steps = fileReader.loadFileHDF5(fileName)
        time = time[start:n_written_steps]-time[start]
        x = x[start:n_written_steps, :]
        y = y[start:n_written_steps, :]
        v_squared = vx[start:n_written_steps, :]**2 + vy[start:n_written_steps, :]**2

        #Q_t = np.zeros([max_steps_tau, n_steps])
        Q_t_time_avg = np.zeros(max_steps_tau)
        Q_t_squared_time_avg = np.zeros(max_steps_tau)
        dE_squared_avg = np.zeros(max_steps_tau)
        dE_quad_avg = np.zeros(max_steps_tau)

        for j in range(0, max_steps_tau):
            #Calc overlap function
            k = tau_integers[j]
            k_max = n_written_steps - start - k
            dx = x[k:k_max+k, :] - x[0:k_max, :]
            dy = y[k:k_max+k, :] - y[0:k_max, :]
            dr = np.sqrt(dx**2+dy**2)
            Q_t = np.mean(np.where(dr<delta,1,0), axis=1)
            Q_t_time_avg[j] = np.mean(Q_t)
            Q_t_squared_time_avg[j] = np.mean(Q_t ** 2)
            #Q_t[j, :] = np.mean(np.where(dr<delta,1,0), axis=1)
            #Q_t[k-1, :] = np.real(np.mean(np.exp(-1j*delta*dr),axis=1))
            #Q_t[k-1, :] = np.mean(np.real(np.exp(-1j * delta * dr)), axis=1)

            #NOTE, 1/2 m has been omitted as it will anyway disappear im the eq. for kurtosis.
            dE_squared = (np.mean(v_squared[k:k_max+k, :] - v_squared[0:k_max, :], axis=1))**2
            dE_squared_avg[j] = np.mean(dE_squared)
            dE_quad_avg[j] = np.mean(dE_squared**2)

        #Q_t_time_avg = np.mean(Q_t, axis=1)
        #Q_t_squared_time_avg = np.mean(Q_t**2, axis=1)

        dt = time[1] - time[0]
        tau = tau_integers*dt
        chi_4 = n_particles*(Q_t_squared_time_avg-Q_t_time_avg**2)
        kurtosis = dE_quad_avg/dE_squared_avg**2

        fileNameOut = fileName + "FpsfAndKurtosis.txt"
        np.savetxt(fileNameOut, np.vstack((tau, Q_t_time_avg, chi_4, kurtosis)).T, fmt="%f")


        h_p_array[i] = np.amax(chi_4)
        u_0_array[i] = u_0

        plt.figure(0)
        #plt.plot(tau, Q_t_time_avg, label=r"$\delta =$ {}".format(delta))
        plt.semilogx(tau, Q_t_time_avg, label=r"$u_0 =$ {}".format(u_0))
        plt.figure(1)
        plt.semilogx(tau, chi_4, label=r"$u_0 =$ {}".format(u_0))
        plt.figure(2)
        plt.loglog(tau, chi_4, label=r"$u_0 =$ {}".format(u_0))
        plt.figure(3)
        plt.semilogx(tau, kurtosis, label=r"$u_0 =$ {}".format(u_0))
        #plt.figure(3)
        #plt.semilogx(tau, strechedExponential(tau, tau_guess, beta_guess, c_guess))


    stop_t = datetime.now()
    print(stop_t-start_t)



    plt.figure(0)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$Q_t$")
    plt.legend()
    plt.figure(1)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$\chi_4$")
    plt.legend()
    plt.figure(2)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$\chi_4$")
    plt.legend()
    plt.figure(3)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$\kappa$")
    plt.legend()

    plt.figure()
    plt.plot(u_0_array, h_p_array)
    plt.xlabel(r"$h_{p}$")
    plt.ylabel(r"$u_{0}$")

    plt.show()


def plotFPSFandKurtisis(fileNameBase):
    n_files = 8
    u_0_array = np.zeros(n_files)
    fileName_array = []
    h_p_array = np.zeros(n_files)
    colourMap = colourSet(n_files)

    for i in range(0, n_files):
        fileName = fileNameBase + str(i)
        fileNameSimPar = fileName + "SimulationParameters.txt"

        R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt = fileReader.loadSimulationParameters(fileNameSimPar)

        u_0_array[i] = u_0
        fileName_array.append(fileName)



    sorted_zip = sorted(zip(u_0_array, fileName_array))
    u_0_sorted = np.zeros(n_files)
    for i in range(len(sorted_zip)):
        tau, Q_t_time_avg, chi_4, kurtosis = fileReader.loadFPSFandKurtosis(sorted_zip[i][1])
        u_0_sorted[i] = sorted_zip[i][0]

        h_p_array[i] = np.amax(chi_4)
        plt.figure(0)
        plt.semilogx(tau, Q_t_time_avg, label=r"$u_0 =$ {}".format(u_0_sorted[i]), color=colourMap[i])
        plt.figure(1)
        plt.semilogx(tau, chi_4, label=r"$u_0 =$ {}".format(u_0_sorted[i]), color=colourMap[i])
        plt.figure(2)
        plt.semilogx(tau, kurtosis, label=r"$u_0 =$ {}".format(u_0_sorted[i]), color=colourMap[i])




    plt.figure(0)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$Q_t$")
    plt.legend()
    plt.savefig(fileNameBase+"_Q_t.png", dpi=200)

    plt.figure(1)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$\chi_4$")
    plt.legend()
    plt.savefig(fileNameBase + "_chi_4.png", dpi=200)

    plt.figure(2)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$\kappa$")
    plt.legend()
    plt.savefig(fileNameBase + "_kappa.png", dpi=200)

    plt.figure()
    plt.scatter(u_0_sorted, h_p_array)
    plt.plot(u_0_sorted, h_p_array)
    plt.xlabel(r"$u_{0}$")
    plt.ylabel(r"$h_{p}$")
    plt.savefig(fileNameBase+"_h_p.png", dpi=200)


    plt.show()


def plotOrderParameterParallell(fileNameBase, n_files):
    u_0_array = np.zeros(n_files)
    fileName_array = []
    colourMap = colourSet(n_files)
    start_file = 0

    for i in range(0, n_files):
        fileName = fileNameBase + str(i+start_file)
        fileNameSimPar = fileName + "SimulationParameters.txt"

        R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt = fileReader.loadSimulationParameters(fileNameSimPar)

        u_0_array[i] = u_0
        fileName_array.append(fileName)

    start = 0#int(20000 / 20)
    sorted_zip = sorted(zip(u_0_array, fileName_array))
    u_0_sorted = np.zeros(n_files)
    for i in range(len(sorted_zip)):
        u_0_sorted[i] = sorted_zip[i][0]
        fileName = sorted_zip[i][1]
        time, x, y, theta, vx, vy, n_particles, n_steps, D_r, u_0, dt, write_interval, n_written_steps = fileReader.loadFileHDF5(fileName)
        time = time[start:n_written_steps]-time[start]
        vx = vx[start:n_written_steps, :]
        vy = vy[start:n_written_steps, :]
        v = np.sqrt(vx**2+vy**2)
        Pi_x = vx/v
        Pi_y = vy/v

        Pi_x_avg = np.mean(Pi_x, axis=1)
        Pi_y_avg = np.mean(Pi_y, axis=1)

        Pi_parallel = np.average(Pi_x*Pi_x_avg[:, None]+Pi_y*Pi_y_avg[:, None], axis=1)

        plt.plot(time, Pi_parallel, label=r"$u_0 =$ {}".format(u_0_sorted[i]), color=colourMap[i])


    plt.legend()
    plt.show()