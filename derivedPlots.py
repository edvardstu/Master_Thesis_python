import numpy as np
from matplotlib import pyplot as plt
import fileReader
import cmath

def plotOrderParameter(time, x, y, theta, vx, vy, D_r):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    #v_0=10
    #beta = np.arctan2(y, x)
    #p_phi = vx/v_0*np.sin(-beta) + vy/v_0*np.cos(-beta)
    #Pi = np.mean(p_phi, axis=1)


    v = np.mean(np.sqrt(vx**2 + vy**2), axis=1)
    beta = np.arctan2(y, x)
    p_phi = vx * np.sin(-beta) + vy * np.cos(-beta)
    Pi = np.mean(p_phi, axis=1)/v


    ax1.plot(time, Pi, color='b')
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$\Pi_\phi$', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2.plot(time, D_r, color='r')
    ax2.set_ylabel(r'Diffusion constant $D_r$', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    plt.savefig("orderParameterPosAvg.png", dpi=200)
    #plt.show()

def plotOrderParameter2(time, x, y, theta, vx, vy, D_r):
    plt.figure()
    #v_0=1
    v = np.mean(np.sqrt(vx**2 + vy**2), axis=1)

    beta = np.arctan2(y, x)
    #p_phi = np.sin(theta - beta)
    #p_phi = vx/v_0*np.sin(-beta) + vy/v_0*np.cos(-beta)
    p_phi = vx * np.sin(-beta) + vy * np.cos(-beta)
    Pi = np.mean(p_phi, axis=1)/v

    plt.plot(time, Pi)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\Pi_\phi$')

    plt.savefig("orderParameterPosAvg2.png", dpi=200)
    #plt.show()


def plotSevralOrderParameterStates(fileNameBase):
    plt.figure()

    n_timeframes = 1
    for i in range (0,4):
        time, x, y, theta, vx, vy, n_particles, n_steps, D_r = loadSeveralFiles(fileNameBase, n_timeframes, i)
        v = np.mean(np.sqrt(vx**2 + vy**2), axis=1)
        beta = np.arctan2(y, x)
        p_phi = vx * np.sin(-beta) + vy * np.cos(-beta)
        Pi = np.mean(p_phi, axis=1)/v

        label = "Not defined"

        if i == 0:
            label = r"$D_r = 0.0$"
        elif i == 1:
            label = r"$D_r = 0.2$"
        elif i == 2:
            label = r"$D_r = 0.8$"
        elif i == 3:
            label = r"$D_r = 5.0$"

        plt.plot(time, np.abs(Pi), label=label)



    plt.xlabel(r'$t$')
    plt.ylabel(r'$|\Pi_\phi|$')
    plt.legend()

    plt.savefig("orderParameterPosAvgSeveral.png", dpi=200)
    plt.show()

def plotAvgV(time, vx, vy):
    v = np.sqrt(vx**2 + vy**2)
    v = np.mean(v, axis=1)
    plt.figure()
    plt.plot(time, v)
    #plt.ylim(0, 6)
    #plt.semilogy(time, v)
    plt.xlabel(r'Time $t$')
    plt.ylabel(r'$\langle v \rangle_{r}$')



def plotTimeAvgOrderParameter(time, x, y, theta, vx, vy, D_r, n_steps, n_timeframes):
    v_0 = 10
    beta = np.arctan2(y, x)
    v = np.mean(np.sqrt(vx**2 + vy**2), axis=1)

    #Pi_r = np.mean(vx/v_0*np.sin(-beta) + vy/v_0*np.cos(-beta), axis=1)
    Pi_r = np.mean(vx*np.sin(-beta) + vy*np.cos(-beta), axis=1)/v



    Pi_rt = np.zeros(n_timeframes)
    #Pi_rt_old = np.zeros(n_timeframes)
    Pi_rt_abs = np.zeros(n_timeframes)
    D_r_rt = np.zeros(n_timeframes)
    Pi_rt_std = np.zeros(n_timeframes)

    start = int(50000/500)

    for i in range(0, n_timeframes):
        Pi_rt[i] = np.mean(Pi_r[(start+n_steps*i):(n_steps*(i+1)-1)])
        Pi_rt_std[i] = np.std(Pi_r[(start+n_steps*i):(n_steps*(i+1)-1)], ddof=1)
        #Pi_rt_old[i] = np.mean(Pi_r_old[(start + n_steps * i):(n_steps * (i + 1) - 1)])
        Pi_rt_abs[i] = np.mean(np.abs(Pi_r[(start + n_steps * i):(n_steps * (i + 1) - 1)]))
        D_r_rt[i] = D_r[start+n_steps*i]

    plt.figure()
    #plt.plot(D_r_rt, Pi_rt)
    plt.errorbar(D_r_rt, Pi_rt, yerr=Pi_rt_std, capsize=5)
    plt.xlabel(r'$D_r$')
    plt.ylabel(r'$\langle \Pi_\phi \rangle_{t}$')
    plt.savefig("orderParameterTimeAvg.png", dpi=200)

    plt.figure()
    plt.errorbar(D_r_rt, Pi_rt_abs, yerr=Pi_rt_std, capsize=5)
    plt.xlabel(r'$D_r$')
    plt.ylabel(r'$\langle|\Pi_\phi|\rangle_{t}$')
    plt.savefig("orderParameterAbsTimeAvg.png", dpi=200)


def plotTimeAvgV(folderName, n_simulations):
    plt.figure()
    n = np.linspace(100, 100 * n_simulations, n_simulations)
    v_avg = np.zeros(n_simulations)

    D_r_values = ['0_0', '0_2', '0_5', '0_8', '1_0']
    for D_r_s in D_r_values:
        for i in range(0, n_simulations):
            #fileName = folderName + 'Dr' + D_r + 'DensitySweep' + str(i) + '.txt'
            fileName = '{}Dr{}DensitySweep{}.txt'.format(folderName, D_r_s, str(i))
            print(fileName)
            time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
            start = int(20000/200)
            v = np.sqrt(vx ** 2 + vy ** 2)
            v = np.mean(v, axis=1)
            v_avg[i] = np.mean(v[start:n_steps])


        plt.plot(n, v_avg, label=r"$D_r$ = {}".format(D_r_s))
    plt.xlabel(r'Number of particles $N$')
    plt.ylabel(r'$\langle v \rangle_{t, r}/v_0$')
    plt.legend()


def plotOrderParameterSqaure(folderName, n_simulations):
    start = int(20000 / 200)
    n = np.linspace(100, 100 * n_simulations, n_simulations)
    Pi_rt = np.zeros(n_simulations)

    D_r_values = ['0_0', '0_2', '0_5', '0_8', '1_0']
    for D_r_s in D_r_values:
        for i in range(n_simulations):
            fileName = '{}Dr{}DensitySweep{}.txt'.format(folderName, D_r_s, str(i))
            print(fileName)
            time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)

            v = np.mean(np.sqrt(vx**2 + vy**2), axis=1)

            Pi_x = vx/v[:, None]
            Pi_y = vy/v[:, None]
            #Pi_x = np.mean(vx, axis=1)/v
            #Pi_y = np.mean(vy, axis=1)/v

            e_r_x = np.mean(Pi_x, axis=1)
            e_r_y = np.mean(Pi_y, axis=1)

            Pi_r = np.mean(Pi_x*e_r_x[:, None] + Pi_y*e_r_y[:, None], axis=1)

            Pi_rt[i] = np.mean(Pi_r[start:n_steps])

        plt.plot(n, Pi_rt, label=r"$D_r$ = {}".format(D_r_s))
    plt.legend()
    plt.xlabel(r'Number of particles $N$')
    plt.ylabel(r'$\Pi_r$')
    #plt.savefig("orderParameterPosAvg2.png", dpi=200)
    #plt.show()


def calcFPSF(fileName):
    time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
    start = int(20000 / 200)

    time = time[start:n_steps]-time[start]
    x = x[start:n_steps, :]
    y = y[start:n_steps, :]
    #dx = x - x[0, :]
    #dy = y - y[0, :]
    #dr = np.sqrt(dx**2 + dy**2)


    #Q_t = np.mean(np.where(dr<delta, 1, 0), axis=1)

    #plt.plot(time, Q_t)
    #plt.show()

    max_steps_tau = 100
    n_steps = n_steps - start - max_steps_tau

    '''dr = np.zeros([max_steps_tau, n_particles])
    for k in range(1, max_steps_tau+1):
        dx = x[k:n_steps, :] - x[0:n_steps-k, :]
        dy = y[k:n_steps, :] - y[0:n_steps - k, :]
        dr[k-1] = np.mean(np.sqrt(dx**2+dy**2), axis=0)'''

    dr = np.zeros([max_steps_tau, n_particles])
    Q_t = np.zeros([max_steps_tau, n_steps])
    #deltas = [0.5, 1, 2, 3, 4, 5, 6, 8, 10]
    deltas = [1., 3., 5., 7., 9., 11.]
    factor = 0.05
    deltas = np.arange(1*factor, 13*factor, 2*factor)
    for delta in deltas:
        for k in range(1, max_steps_tau+1):
            dx = x[k:n_steps+k, :] - x[0:n_steps, :]
            dy = y[k:n_steps+k, :] - y[0:n_steps, :]
            dr = np.sqrt(dx**2+dy**2)
            #Q_t[k-1, :] = np.mean(np.where(dr<delta,1,0), axis=1)
            #test1 = np.mean(np.where(dr<delta,1,0), axis=1)
            #test2 = np.mean(np.abs(np.real(np.exp(-1j*delta*dr))),axis=1)
            #plt.plot(test1)
            #plt.plot(test2)
            #plt.show()
            Q_t[k-1, :] = np.mean(np.real(np.exp(-1j*delta*dr)),axis=1)

        Q_t_time_avg = np.mean(Q_t, axis=1)
        Q_t_squared_time_avg = np.mean(Q_t**2, axis=1)

        tau = np.linspace(1, max_steps_tau + 1, max_steps_tau)
        chi_4 = n_particles*(Q_t_squared_time_avg-Q_t_time_avg**2)

        plt.figure(0)
        plt.semilogx(tau, Q_t_time_avg, label=r"$\delta =$ {}".format(delta))
        plt.figure(1)
        plt.semilogx(tau, chi_4, label=r"$\delta =$ {}".format(delta))
        plt.figure(2)
        plt.loglog(tau, chi_4, label=r"$\delta =$ {}".format(delta))

    plt.figure(0)
    plt.legend()
    plt.figure(1)
    plt.legend()
    plt.figure(2)
    plt.legend()
    plt.show()
    '''
    deltas = [0.5, 1, 3, 5, 8, 10]
    for delta in deltas:
        Q_t = np.mean(np.where(dr<delta,1,0), axis=1)

        plt.plot(tau,Q_t, label=r"$\delta =$ {}".format(delta))
    plt.legend()
    #plt.show()
    '''