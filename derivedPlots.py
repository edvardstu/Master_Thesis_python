import numpy as np
from matplotlib import pyplot as plt

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
    plt.ylim(0, 6)
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