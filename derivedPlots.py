import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import fileReader
import math
from scipy import stats
from datetime import datetime
import os.path
import tikzplotlib

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

def plotAvgEnergy(fileName):
    time, x, y, theta, vx, vy, n_particles, n_steps, D_r, u_0, dt, write_interval, n_written_steps, L, H = fileReader.loadFileHDF5(fileName)
    E_0 = u_0**2
    start = 4000#int(20000 / 200)
    n_steps = len(time)
    time = time[start:n_steps]
    vx = vx[start:n_steps, :]
    vy = vy[start:n_steps, :]
    E = np.mean(vx**2+vy**2, axis=1)
    plt.figure()
    plt.plot(time, E/E_0)
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'$\langle E \rangle_{r}/E_{0}$')
    plt.savefig(fileName+"_E_t.png", dpi=200)
    tikzplotlib.save(fileName+"_E_t.tex")


    n_bins = 24
    plt.figure()
    n, bins, patches = plt.hist(np.log10(E/E_0), n_bins, facecolor='blue', alpha=0.5)
    bins = (bins[0:n_bins]+bins[1:n_bins+1])/2
    plt.xlabel(r"Magnitude $M$")
    plt.ylabel(r"# Occurrences, $N_{m}$")
    plt.savefig(fileName+"_Histogram.png", dpi=200)
    tikzplotlib.save(fileName+"_Histogram.tex")

    fileNameOut = fileName + "Histogram.txt"
    np.savetxt(fileNameOut, np.vstack((n, bins)).T, fmt="%f")


def plotHistogram(fileNameBase):
    n, magnitude = fileReader.loadHistogram(fileNameBase)
    plt.figure()
    plt.scatter(magnitude, n, label="Data points")
    plt.yscale('log')


    start = np.argmax(n)
    end = len(n)

    n = n[start:end]
    magnitude = magnitude[start:end]
    slope, intercept, r_value, p_value, std_err = stats.linregress(magnitude, np.log10(n))
    magnitude_line=np.zeros(2)
    magnitude_line[0]=magnitude[0]
    magnitude_line[1]=magnitude[-1]
    n_line= 10**(intercept+slope*magnitude_line)
    #plt.figure()
    #plt.plot(magnitude, np.log10(n))
    label_reg = r"$10^{a-bM}$, $a=$%.2f, $b=$%.2f" % (intercept, slope)

    plt.plot(magnitude_line, n_line, label=label_reg)
    plt.legend()
    plt.xlabel(r"Magnitude $M$")
    plt.ylabel(r"# Occurrences, $N_{m}$")
    #plt.savefig(fileNameBase+"_GR.png", dpi=200)
    #tikzplotlib.save(fileNameBase+"_GR.tex")
    print(intercept, slope, std_err)


def plotSeveralHistograms(fileNameBase, n_files):
    u_0_array = np.zeros(n_files)
    fileName_array = []

    for i in range(0, n_files):
        fileName = fileNameBase + str(i)
        fileNameSimPar = fileName + "SimulationParameters.txt"

        R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt = fileReader.loadSimulationParameters(fileNameSimPar)

        u_0_array[i] = u_0
        fileName_array.append(fileName)

    sorted_zip = sorted(zip(u_0_array, fileName_array))
    u_0_sorted = np.zeros(n_files)

    rows = int(np.ceil(n_files/2))
    fig, axs = plt.subplots(rows, 2)

    for i in range(len(sorted_zip)):
        u_0_sorted[i] = sorted_zip[i][0]
        n, magnitude = fileReader.loadHistogram(sorted_zip[i][1])
        u_0_sorted[i] = sorted_zip[i][0]

        col = int(i%2)
        row = int((i-col)/2)
        axs[row, col].bar(magnitude, n, magnitude[1]-magnitude[0])
        axs[row, col].set_title(r'$u_0=$%.1f' % u_0_sorted[i])

        #9axs[row, col].yscale('log')
        #plt.yscale('log')
        #plt.xscale('log')


        #if col == 0:
        #    axs[row, col].set(ylabel=r"# Occurrences, $N_{m}$")
        #if row == rows-1:
        #    axs[row, col].set(xlabel=r"Magnitude $M$")

        axs[row, col].set(yscale = "log")

    #for ax in axs.flat:
        #ax.set(xlabel=r"Magnitude $M$", ylabel=r"# Occurrences, $N_{m}$", yscale='log')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
        #ax.label_outer()

    plt.savefig(fileNameBase+"_Histograms.png", dpi=200)
    tikzplotlib.save(fileNameBase+"_Histograms.tex")







def calcFPSFandKurtisis(fileNameBase, n_files, start_file):
    start_t = datetime.now()
    start = 4000#int(20000 / 20)
    max_steps_tau = 300
    steps_lin = np.floor(max_steps_tau/2)
    steps_log = np.ceil(max_steps_tau/2)
    tau_temp_1 = np.linspace(1, steps_lin, steps_lin, dtype=int)
    tau_temp_2 = np.geomspace(steps_lin+1, 15000, steps_log, dtype=int)
    tau_integers = np.concatenate((tau_temp_1, tau_temp_2))

    delta = 0.3

    h_p_array = np.zeros(n_files)
    u_0_array = np.zeros(n_files)
    for i in range(0, n_files):
        fileName = fileNameBase + str(i+start_file)# + ".h5"
        #time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
        time, x, y, theta, vx, vy, n_particles, n_steps, D_r, u_0, dt, write_interval, n_written_steps, L, H = fileReader.loadFileHDF5(fileName)
        del theta
        dt = time[1] - time[0]
        del time

        x = x[start:n_written_steps, :]
        y = y[start:n_written_steps, :]
        v_squared = vx[start:n_written_steps, :]**2 + vy[start:n_written_steps, :]**2
        del vx
        del vy

        #Q_t = np.zeros([max_steps_tau, n_steps])
        Q_t_time_avg = np.zeros(max_steps_tau)
        Q_t_squared_time_avg = np.zeros(max_steps_tau)
        dE_squared_avg = np.zeros(max_steps_tau)
        dE_quad_avg = np.zeros(max_steps_tau)

        for j in range(0, max_steps_tau):
            if ((j+1)%3)==0:
                print((j+1)/max_steps_tau)
            #Calc overlap function
            k = tau_integers[j]
            k_max = n_written_steps - start - k
            dx = np.abs(x[k:k_max+k, :] - x[0:k_max, :])
            dx = np.fmin(dx, L-dx)
            dy = np.abs(y[k:k_max+k, :] - y[0:k_max, :])
            dy = np.fmin(dy, H-dy)
            #dr = np.sqrt(dx**2+dy**2)
            dr = np.sqrt(dx ** 2 + dy ** 2)
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


        tau = tau_integers*dt
        chi_4 = n_particles*(Q_t_squared_time_avg-Q_t_time_avg**2)

        kurtosis = dE_quad_avg/dE_squared_avg**2

        fileNameOut = fileName + "FpsfAndKurtosis.txt"
        np.savetxt(fileNameOut, np.vstack((tau, Q_t_time_avg, chi_4, kurtosis)).T, fmt="%f")

    end_t = datetime.now()

    print("Time use: {}".format(end_t-start_t))


def plotFPSFandKurtisis(fileNameBase, n_files):
    u_0_array = np.zeros(n_files)
    fileName_array = []
    h_p_array = np.zeros(n_files)
    kappa_ex_array = np.zeros(n_files)
    colourMap = colourSet(n_files)

    for i in range(0, n_files):
        fileName = fileNameBase + str(i)
        fileNameSimPar = fileName + "SimulationParameters.txt"

        R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt = fileReader.loadSimulationParameters(fileNameSimPar)

        u_0_array[i] = u_0
        fileName_array.append(fileName)


    #plt.figure(0, figsize=(4,3))


    sorted_zip = sorted(zip(u_0_array, fileName_array))
    u_0_sorted = np.zeros(n_files)
    for i in range(len(sorted_zip)):
        tau, Q_t_time_avg, chi_4, kurtosis = fileReader.loadFPSFandKurtosis(sorted_zip[i][1])
        u_0_sorted[i] = sorted_zip[i][0]

        h_p_array[i] = np.amax(chi_4)
        kappa_ex_array[i] = kurtosis[0]-3.0
        plt.figure(0)
        plt.semilogx(tau, Q_t_time_avg, label=r"$u_0 =$ {}".format(u_0_sorted[i]), color=colourMap[i])
        plt.figure(1)
        plt.semilogx(tau, chi_4, label=r"$u_0 =$ {}".format(u_0_sorted[i]), color=colourMap[i])
        plt.figure(2)
        plt.loglog(tau, kurtosis, label=r"$u_0 =$ {}".format(u_0_sorted[i]), color=colourMap[i])
        #plt.scatter(tau, kurtosis, label=r"$u_0 =$ {}".format(u_0_sorted[i]), color=colourMap[i])

        #plt.yscale('log')
        #plt.xscale('log')




    plt.figure(0)
    #plt.margins(1.0)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$Q_t$")
    plt.legend()
    plt.savefig(fileNameBase+"_Q_t.png", dpi=200)
    tikzplotlib.save(fileNameBase+"_Q_t.tex")

    plt.figure(1)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$\chi_4$")
    plt.legend()
    plt.savefig(fileNameBase + "_chi_4.png", dpi=200)
    tikzplotlib.save(fileNameBase + "_chi_4.tex")

    plt.figure(2)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$\kappa$")
    plt.legend()
    plt.savefig(fileNameBase + "_kappa.png", dpi=200)
    tikzplotlib.save(fileNameBase + "_kappa.tex")

    plt.figure()
    plt.scatter(u_0_sorted, h_p_array)
    plt.plot(u_0_sorted, h_p_array)
    plt.xlabel(r"$u_{0}$")
    plt.ylabel(r"$h_{p}$")
    plt.savefig(fileNameBase+"_h_p.png", dpi=200)
    tikzplotlib.save(fileNameBase + "_h_p.tex")

    plt.figure()
    plt.scatter(u_0_sorted, kappa_ex_array)
    plt.plot(u_0_sorted, kappa_ex_array)
    plt.xlabel(r"$u_{0}$")
    plt.ylabel(r"$\kappa_{ex}$")
    plt.yscale('log')
    plt.savefig(fileNameBase + "_kappa_ex.png", dpi=200)
    tikzplotlib.save(fileNameBase + "_kappa_ex.tex")

    #plt.rcParams.update({'font.size': 30})
    plt.show()

def plotFPSPwithRegression(fileNameBase, u_0_ref, n_files):
    for i in range(0, n_files):
        fileName = fileNameBase + str(i)
        fileNameSimPar = fileName + "SimulationParameters.txt"

        R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt = fileReader.loadSimulationParameters(fileNameSimPar)
        if(u_0_ref==u_0):
            print(fileName)
            break

    tau, Q_t_time_avg, chi_4, kurtosis = fileReader.loadFPSFandKurtosis(fileName)
    plt.figure(0)
    plt.plot(tau, chi_4,label="Data points")
    start = 14
    stop =147
    tau = tau[start:stop]
    chi_4 = chi_4[start:stop]
    plt.scatter(tau, chi_4, marker="+")
    log_tau = np.log10(tau)
    log_chi_4 = np.log10(chi_4)

    plt.figure(1)
    plt.plot(log_tau, log_chi_4, label="Data points")
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau, log_chi_4)
    log_reg = intercept+slope*log_tau
    label_reg = r"$a=$%.2f, $b=$%.2f" % (intercept, slope)
    plt.plot(log_tau, log_reg, label=label_reg)
    plt.legend()

    plt.figure(0)
    plt.plot(tau, 10**log_reg, label=label_reg)
    plt.xscale("log")
    plt.yscale("log")

    plt.legend()
    plt.show()

def plotFPSFandKurtisisSeveral(folderName, fileName, n_files):
    u_0_array = np.zeros(n_files)
    u_0_ref = np.zeros(n_files)
    u_0_sorted = np.zeros(n_files)
    fileName_array = []

    h_p_array = np.zeros(n_files)
    colourMap = colourSet(n_files)

    subfolders = [f.name for f in os.scandir(folderName) if f.is_dir()]

    for i_folder, subfolder in enumerate(subfolders):
        fileName_array_temp = []
        fileName_array.append([])
        fileNameBase = folderName + subfolder + "/" + fileName
        for i in range(0, n_files):
            fileNameSimPar = fileNameBase+ str(i) + "SimulationParameters.txt"

            R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt = fileReader.loadSimulationParameters(fileNameSimPar)

            u_0_array[i] = u_0
            fileName_array_temp.append(fileNameBase + str(i))

        sorted_zip = sorted(zip(u_0_array, fileName_array_temp))

        if i_folder==0:
            for i in range(len(sorted_zip)):
                u_0_ref[i] = sorted_zip[i][0]
        else:
            for i in range(len(sorted_zip)):
                u_0_sorted[i] = sorted_zip[i][0]
            if np.all(u_0_ref != u_0_sorted):
                print("Folders not containing same u_0 values")
                print("Referecne subfolder has u_0:")
                print(u_0_ref)
                print("Subfolder " + subfolder + " has u_0:")
                print(u_0_sorted)
        for i in range(len(sorted_zip)):
            fileName_array[i_folder].append(sorted_zip[i][1])
    print(fileName_array)

    n_points=300
    n_subfolders = len(subfolders)
    tau = np.zeros(n_points)
    #Q_t_time_avg = np.zeros((n_points, n_subfolders))
    #chi_4 = np.zeros((n_points, n_subfolders))
    #kurtosis = np.zeros((n_points, n_subfolders))

    for i in range(n_files):
        Q_t_time_avg = np.zeros((n_subfolders, n_points))
        chi_4 = np.zeros((n_subfolders, n_points))
        kurtosis = np.zeros((n_subfolders, n_points))
        for j in range(n_subfolders):
            #print(fileName_array[j][i])
            tau_temp, Q_t_time_avg_temp, chi_4_temp, kurtosis_temp = fileReader.loadFPSFandKurtosis(fileName_array[j][i])
            if j==0:
                tau = tau_temp
            Q_t_time_avg[j] = Q_t_time_avg_temp
            chi_4[j] = chi_4_temp
            kurtosis[j] = kurtosis_temp

        Q_t_time_avg = np.mean(Q_t_time_avg, axis=0)
        chi_4 = np.mean(chi_4, axis=0)
        kurtosis = np.mean(kurtosis, axis=0)

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
    plt.savefig(folderName + fileName + "_Q_t.png", dpi=200)

    plt.figure(1)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$\chi_4$")
    plt.legend()
    plt.savefig(folderName + fileName + "_chi_4.png", dpi=200)

    plt.figure(2)
    plt.xlabel(r"Time $\tau$")
    plt.ylabel(r"$\kappa$")
    plt.legend()
    plt.savefig(folderName + fileName + "_kappa.png", dpi=200)

    plt.figure()
    plt.scatter(u_0_sorted, h_p_array)
    plt.plot(u_0_sorted, h_p_array)
    plt.xlabel(r"$u_{0}$")
    plt.ylabel(r"$h_{p}$")
    plt.savefig(folderName+ fileName + "_h_p.png", dpi=200)

    plt.show()


def plotOrderParameterParallell(fileNameBase, n_files):
    u_0_array = np.zeros(n_files)
    Pi_parallel_time_average = np.zeros(n_files)
    Pi_parallel_time_average_std = np.zeros(n_files)
    fileName_array = []
    colourMap = colourSet(n_files)
    start_file = 0

    for i in range(0, n_files):
        fileName = fileNameBase + str(i+start_file)

        fileNameSimPar = fileName + "SimulationParameters.txt"

        R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt = fileReader.loadSimulationParameters(fileNameSimPar)

        u_0_array[i] = u_0
        fileName_array.append(fileName)

    start = 10000#int(20000 / 20)
    sorted_zip = sorted(zip(u_0_array, fileName_array))
    u_0_sorted = np.zeros(n_files)
    for i in range(len(sorted_zip)):
        u_0_sorted[i] = sorted_zip[i][0]
        fileName = sorted_zip[i][1]
        time, x, y, theta, vx, vy, n_particles, n_steps, D_r, u_0, dt, write_interval, n_written_steps, L, H = fileReader.loadFileHDF5(fileName)
        del x
        del y
        del theta
        del time
        print(fileName)
        #time = time[start:n_written_steps]#-time[start]
        vx = vx[start:n_written_steps, :]
        vy = vy[start:n_written_steps, :]
        v = np.sqrt(vx**2+vy**2)

        Pi_x = vx/v
        Pi_y = vy/v
        del vx
        del vy
        del v

        Pi_x_avg = np.mean(Pi_x, axis=1)
        Pi_y_avg = np.mean(Pi_y, axis=1)


        Pi_parallel = np.mean(np.abs(Pi_x*Pi_x_avg[:, None]+Pi_y*Pi_y_avg[:, None]), axis=1)
        del Pi_x
        del Pi_y
        del Pi_x_avg
        del Pi_y_avg
        where_are_NaNs = np.isnan(Pi_parallel)
        Pi_parallel[where_are_NaNs] = 0.0
        del where_are_NaNs

        #plt.plot(time, Pi_parallel, label=r"$u_0 =$ {}".format(u_0_sorted[i]), color=colourMap[i])

        Pi_parallel_time_average[i] = np.mean(Pi_parallel)
        Pi_parallel_time_average_std[i] = np.std(Pi_parallel, ddof=1)

    #plt.xlabel(r"$t$")
    #plt.ylabel(r"$\Pi_{||}(t)$")
    #plt.savefig(fileNameBase+"_Pi_par.png", dpi=200)
    #plt.legend()

    print(u_0_sorted)
    print(Pi_parallel_time_average)
    print(Pi_parallel_time_average_std)

    fileNameOut = fileNameBase + "OrderParameter.txt"
    np.savetxt(fileNameOut, np.vstack((u_0_sorted, Pi_parallel_time_average, Pi_parallel_time_average_std)).T, fmt="%f")

    plt.figure()
    plt.errorbar(u_0_sorted, Pi_parallel_time_average, yerr=Pi_parallel_time_average_std, capsize=5)
    plt.xlabel(r"$u_{0}$")
    plt.ylabel(r"$\langle \Pi_{||}(u_{0}) \rangle_{t}$")
    plt.savefig(fileNameBase+"_Pi_par_avg.png", dpi=200)
    tikzplotlib.save(fileNameBase + "_Pi_par_avg.tex")
    plt.show()


def findFileName(fileNameBase, u_0_find):
    exists = True
    i=0
    while exists:
        fileName = fileNameBase + str(i) + "SimulationParameters.txt"
        print(fileName)
        if (os.path.isfile(fileName)):
            R, L, H, h, r_a, n_particles, n_fixed_particles, u_0, D_r, n_steps, dt = fileReader.loadSimulationParameters(fileName)
            print(r"The file number for u_0={} is {}".format(u_0, i))
            if u_0_find == u_0:
                print(r"The file number for u_0={} is {}".format(u_0_find, i))
                return fileName
            i+=1
        else:
            print("File does not exists")
            exists = False

    return None