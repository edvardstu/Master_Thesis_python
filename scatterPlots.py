import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import enum
import tikzplotlib
import fileReader

class Barrier(enum.Enum):
    Circular = 1
    Periodic = 2
    PeriodicTube = 3
    PeriodicFunnel = 4

def plotBoundary(barrier, L, H, h, R):
    boundary = np.array([None, None])
    boundary_periodic = np.array([None, None])
    if barrier == Barrier.Circular:
        n_points = 100
        d_theta = 2 * np.pi / 100
        theta1 = np.linspace(0, 2 * np.pi - d_theta, n_points)
        theta2 = np.linspace(d_theta, 2 * np.pi, n_points)
        boundary = np.array([R * np.cos(theta1), R * np.cos(theta2), R * np.sin(theta1), R * np.sin(theta2)])
    elif barrier == Barrier.Periodic:
        boundary_periodic = np.array([[-L / 2, -L / 2, L / 2, -L / 2],
                                      [L / 2, L / 2, L / 2, -L / 2],
                                      [H / 2, -H / 2, -H / 2, -H / 2],
                                      [H / 2, -H / 2, H / 2, H / 2]])
    elif barrier == Barrier.PeriodicTube:
        boundary = np.array([[-L / 2, -L / 2],
                             [L / 2, L / 2],
                             [H / 2, -H / 2],
                             [H / 2, -H / 2]])
        boundary_periodic = np.array([[L / 2, -L / 2],
                                      [L / 2, -L / 2],
                                      [H / 2, H / 2],
                                      [-H / 2, -H / 2]])
    elif barrier == Barrier.PeriodicFunnel:
        boundary = np.array([[-L / 2, 0, -L / 2, 0],
                             [0, L / 2, 0, L / 2],
                             [H / 2, h / 2, -H / 2, -h / 2],
                             [h / 2, H / 2, -h / 2, -H / 2]])
        boundary_periodic = np.array([[L / 2, -L / 2],
                                      [L / 2, -L / 2],
                                      [H / 2, H / 2],
                                      [-H / 2, -H / 2]])

    if (boundary.any() != None):
        n_boundary = len(boundary[0])
        for i in range(0, n_boundary):
            plt.plot([boundary[0, i], boundary[1, i]], [boundary[2, i], boundary[3, i]], color='black')

    if (boundary_periodic.any() != None):
        n_boundary_periodic = len(boundary_periodic[0])
        for i in range(0, n_boundary_periodic):
            plt.plot([boundary_periodic[0, i], boundary_periodic[1, i]],
                     [boundary_periodic[2, i], boundary_periodic[3, i]], color='black', linestyle='dashed')

def plotLimits(barrier, L, H, R):
    plt.axis('scaled')
    if barrier == Barrier.Circular:
        plt.xlim(-R - R/10, R + R/10)
        plt.ylim(-R - R/10, R + R/10)
    else:
        plt.xlim(-L/2 - L / 20, L/2 + L / 20)
        plt.ylim(-H/2 - H / 20, H/2 + H / 20)

def plotState(x, y, vx, vy, i):
    plt.figure()
    x_0 = x[i]
    y_0 = y[i]
    c = np.arctan2(vy[i], vx[i])
    x_0 = np.append(x_0, [100, 100])
    y_0 = np.append(y_0, [100, 100])
    c = np.append(c, [-np.pi, np.pi])
    #c = np.log(vx[i]**2+vy[i]**2)
    scat = plt.scatter(x_0, y_0, c=c, cmap='hsv')
    plt.quiver(x_0, y_0, np.cos(c), np.sin(c))

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    L=25
    H=25
    plt.xlim(-L / 2 - L / 20, L / 2 + L / 20)
    plt.ylim(-H / 2 - H / 20, H / 2 + H / 20)


    cbar = plt.colorbar(scat)#, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar.ax.set_ylabel(r'$\theta$')
    #cbar.set_clim(-np.pi, np.pi)
    #plt.clim(-np.pi, np.pi)


def plotStateAndSaveFile(fileName, frame):
    time, x, y, theta, vx, vy, n_particles, n_steps, D_r, u_0, dt, write_interval, n_written_steps, L, H = fileReader.loadFileHDF5(fileName)
    plt.figure()
    x_0 = x[frame]
    y_0 = y[frame]
    c = np.arctan2(vy[frame], vx[frame])
    #x_0 = np.append(x_0, [100, 100])
    #y_0 = np.append(y_0, [100, 100])
    #c = np.append(c, [-np.pi, np.pi])

    scat = plt.scatter(x_0, y_0, c=c, cmap='hsv')
    #v=np.cos(c)
    #u=np.sin(c)
    v= np.abs(vx[frame])/u_0
    u= np.abs(vy[frame])/u_0
    plt.quiver(x_0, y_0, v, u)

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    plt.xlim(-L / 2 - L / 20, L / 2 + L / 20)
    plt.ylim(-H / 2 - H / 20, H / 2 + H / 20)

    cbar = plt.colorbar(scat)  # , ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar.ax.set_ylabel(r'$\theta$')
    plt.clim(-np.pi, np.pi)
    plt.savefig(fileName + "_state_" + str(frame) + ".png", dpi=200)
    fileNameOut = fileName + "_state_" + str(frame) + ".txt"
    np.savetxt(fileNameOut, np.vstack((x_0, y_0, c, v, u)).T, fmt="%f")


def plotIntermittency(fileName, frame, extra, steps):
    time, x, y, theta, vx, vy, n_particles, n_steps, D_r, u_0, dt, write_interval, n_written_steps, L, H = fileReader.loadFileHDF5(fileName)

    for i in range(frame-extra*steps, frame+extra*steps+1,steps):
        plt.figure()
        plt.title("Time = %.2f" % time[i])
        x_0 = x[i]
        y_0 = y[i]
        c = np.log10((vx[i]**2+vy[i]**2)/u_0**2)
        scat = plt.scatter(x_0, y_0, c=c, cmap='plasma')
        #plt.quiver(x_0, y_0, np.cos(c), np.sin(c))

        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')

        plt.clim(-4, 0)
        cbar = plt.colorbar(scat)
        cbar.ax.set_ylabel(r'$E_i/E_0$')
        barrier = Barrier.Periodic
        plotBoundary(barrier, L, H, 0, 0)
        plt.savefig(fileName + "_intermittency_" + str(i) + ".png", dpi=200)
        tikzplotlib.save(fileName + "_intermittency_" + str(i) + ".tex")

def plotPiNematic(fileName, frame):
    time, x, y, theta, vx, vy, n_particles, n_steps, D_r, u_0, dt, write_interval, n_written_steps, L, H = fileReader.loadFileHDF5(fileName)

    plt.figure()
    plt.title("Time = %.2f" % time[frame])
    x_0 = x[frame]
    y_0 = y[frame]
    theta_0 = theta[frame]
    u = np.cos(theta_0)
    v = np.sin(theta_0)

    theta_0 = np.arctan2(v,u)
    #theta_0 = np.where(theta_0 > np.pi / 2, theta_0-np.pi, theta_0)
    #theta_0 = np.where(theta_0 < -np.pi / 2, theta_0 + np.pi, theta_0)

    scat = plt.scatter(x_0, y_0, c=theta_0, cmap="hsv")
    plt.quiver(x_0, y_0, u, v)

    #c = np.arctan2(vy[i], vx[i])
    #x_0 = np.append(x_0, [100, 100])
    #y_0 = np.append(y_0, [100, 100])
    #c = np.append(c, [-np.pi, np.pi])
    #c = np.log(vx[i]**2+vy[i]**2)
    #scat = plt.scatter(x_0, y_0, c=c, cmap='hsv')

    L = 25
    H = 25
    plt.xlim(-L / 2 - L / 20, L / 2 + L / 20)
    plt.ylim(-H / 2 - H / 20, H / 2 + H / 20)

    cbar = plt.colorbar(scat)  # , ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar.ax.set_ylabel(r'$\theta$')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    barrier = Barrier.Periodic
    plotBoundary(barrier, L, H, 0, 0)
    plt.savefig(fileName + "_Pi_nem_new_" + str(frame) + ".png", dpi=200)
    tikzplotlib.save(fileName + "_Pi_nem_new_" + str(frame) + ".tex")
    fileNameOut = fileName + "_Pi_nem_new_" + str(frame) + ".txt"
    np.savetxt(fileNameOut, np.vstack((x_0, y_0, theta_0, u, v)).T, fmt="%f")

    size_l= 5
    indecies = (abs(x_0)<=size_l) & (abs(y_0)<=size_l)

    length = np.sum(indecies)
    x_new = np.zeros(length)
    y_new = np.zeros(length)
    t_new = np.zeros(length)
    u_new = np.zeros(length)
    v_new = np.zeros(length)

    j=0
    print(length)
    for i in range(1000):
        if indecies[i]==1:
            x_new[j] = x_0[i]
            y_new[j] = y_0[i]
            t_new[j] = theta_0[i]
            u_new[j] = u[i]
            v_new[j] = v[i]
            j+=1

    fileNameOut = fileName + "_Pi_nem_new_short_" + str(frame) + ".txt"
    np.savetxt(fileNameOut, np.vstack((x_new, y_new, t_new, u_new, v_new)).T, fmt="%f")
    plt.figure()
    plt.scatter(x_new, y_new)

def plotVelocityMap(x, y, vx, vy, i, R):
    plt.figure()
    x = x[i]
    y = y[i]
    v = np.sqrt(vx[i]**2+vy[i]**2)
    #v = np.where(v > 10, 10, v)
    scat = plt.scatter(x, y, c=v, cmap='Spectral')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    cbar = plt.colorbar(scat)
    cbar.ax.set_ylabel(r'$v(t)$')
    cbar.set_clim(-2*np.pi, 2*np.pi)

    # Add confining circle
    theta_circle = np.linspace(0, 2 * np.pi, 1500)
    plt.plot(R * np.cos(theta_circle), R * np.sin(theta_circle), color=(0, 0, 0, 0.5))

    plt.axis('scaled')
    plt.xlim(-19, 19)
    plt.ylim(-19, 19)
    plt.savefig("velocityMapDisordered.png", dpi=200)


def plotDeformationMap(x, y, deformation, i, R):
    plt.figure()
    x = x[i]
    y = y[i]
    deformation = deformation[i]
    scat = plt.scatter(x, y, c=deformation, cmap='coolwarm')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    cbar = plt.colorbar(scat)
    cbar.ax.set_ylabel(r'$v(t)$')

    # Add confining circle
    theta_circle = np.linspace(0, 2 * np.pi, 1500)
    plt.plot(R * np.cos(theta_circle), R * np.sin(theta_circle), color=(0, 0, 0, 0.5))

    plt.axis('scaled')
    plt.xlim(-19, 19)
    plt.ylim(-19, 19)
    plt.savefig("velocityMapDisordered.png", dpi=200)


def plotParticleTrace(x, y, numParticles, R):
    plt.figure()
    start = int(50000/100)
    stop = 1500

    for i in range(0, numParticles):
        plt.plot(x[start:stop, i], y[start:stop, i])


    # Add confining circle
    theta_circle = np.linspace(0, 2 * np.pi, 1500)
    plt.plot(R * np.cos(theta_circle), R * np.sin(theta_circle), color=(0, 0, 0, 0.5))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    #plt.axis('equal')
    plt.axis('scaled')
    plt.xlim(-18, 18)
    plt.ylim(-18, 18)
    plt.savefig("particleTraceDisordered.png", dpi=200)


def run_animation_old(x, y, theta, R, n_steps, n_particles, n_frames):
    r_c = 1 / 2

    xy = np.zeros([n_frames, n_particles, 2])
    theta_animation = np.zeros([n_frames, n_particles])
    for i in range(0, n_frames):
        offset = int(n_steps / n_frames)
        xy[i] = np.stack((x[i * offset].T, y[i * offset].T)).T
        theta_animation[i] = theta[i * offset]

    numframes = n_frames
    numpoints = n_particles

    x_0 = x[0]
    y_0 = y[0]
    c = np.arctan2(y[1] - y[0], x[1] - x[0])

    fig = plt.figure(facecolor='white', figsize=(9, 7))
    ax = fig.add_subplot(111, aspect='equal')

    # plt.xlim(-R - R/10, R + R/10)
    # plt.ylim(-R - R/10, R + R/10)
    # Set limits
    ax.axis([-R - R / 10, R + R / 10, -R - R / 10, R + R / 10])
    # ax.axis([-R *10, R *10, -R *10, R*10])
    # ax.axis([-2, 18, 0, 45])

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Add confining circle
    theta_circle = np.linspace(0, 2 * np.pi, 1000)
    plt.plot(R * np.cos(theta_circle), R * np.sin(theta_circle), color=(0, 0, 0, 0.5))

    # scat = ax.scatter(x, y, s=0, alpha=0.5, clip_on=False)
    scat = ax.scatter(x_0, y_0, c=c, cmap='hsv')

    rpix = 6  # 12#25
    # Calculate and update size in points:
    size_pt = (2 * rpix / fig.dpi * 72) ** 2
    sizes_pt = np.full(n_particles, size_pt)
    scat.set_sizes(sizes_pt)

    arrow = plt.quiver(x_0, y_0, np.cos(c), np.sin(c))

    cbar = fig.colorbar(scat, ax=ax)
    cbar.ax.set_ylabel(r'$\theta$')
    # cbar.set_clim(-2*np.pi, 2*np.pi)
    ani = animation.FuncAnimation(fig, update_plot, frames=numframes,
                                  fargs=(xy, scat, arrow))
    #ani.save('testAB.gif', writer='imagemagick', fps=5)
    plt.show()


def run_animation(x, y, theta, vx, vy, L, H, h, R, n_steps, n_particles, n_frames, barrier):
    r_c = 1 / 2

    xy = np.zeros([n_frames, n_particles, 2])
    theta_animation = np.zeros([n_frames, n_particles])
    vxy = np.zeros([n_frames, n_particles, 2])
    for i in range(0, n_frames):
        offset = int(n_steps / n_frames)
        xy[i] = np.stack((x[i * offset].T, y[i * offset].T)).T
        theta_animation[i] = theta[i * offset]
        #vxy[i] = np.stack((vx[i * offset].T, vy[i * offset].T)).T
        vxy[i] = np.stack((np.cos(theta[i * offset].T), np.sin(theta[i * offset].T))).T

    numframes = n_frames
    numpoints = n_particles

    x_0 = x[0]
    y_0 = y[0]
    #c = np.zeros(1000)
    c = np.arctan2(y[1] - y[0], x[1] - x[0])
    #c = np.log(np.sqrt(vxy[0, :, 1] ** 2 + vxy[0, :, 0] ** 2))

    fig = plt.figure(facecolor='white', figsize=(7, 5))
    ax = fig.add_subplot(111, aspect='equal')
    plt.tight_layout(pad=3)

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    plotBoundary(barrier, L, H, h, R)
    plotLimits(barrier, L, H, R)

    # scat = ax.scatter(x, y, s=0, alpha=0.5, clip_on=False)
    #scat = ax.scatter(x_0, y_0, c=c, cmap='hsv')
    scat = ax.scatter(x_0, y_0, c=c, cmap='hsv')

    #rpix = 6  # 12#25
    rpix = 3
    # Calculate and update size in points:
    size_pt = (2 * rpix / fig.dpi * 72) ** 2
    sizes_pt = np.full(n_particles, size_pt)
    scat.set_sizes(sizes_pt)

    arrow = plt.quiver(x_0, y_0, np.cos(c), np.sin(c))

    cbar = fig.colorbar(scat, ax=ax)
    cbar.ax.set_ylabel(r'$\theta$')
    # cbar.set_clim(-2*np.pi, 2*np.pi)
    ani = animation.FuncAnimation(fig, update_plot, frames=numframes,
                                  fargs=(xy, vxy, scat, arrow))
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #ani.save('test.mp4', writer=writer)
    #ani.save('test.gif', writer='imagemagick', fps=5)
    plt.show()


def update_plot(i, data, vxy, scat, arrow):
    #cm = np.arctan2(data[i, :, 1] - data[i - 1, :, 1], data[i, :, 0] - data[i - 1, :, 0])
    cm = np.zeros(len(vxy[i, :, 1]))
    #cm = np.arctan2(vxy[i, :, 1], vxy[i, :, 0])

    scat.set_offsets(data[i])
    scat.set_array(cm)
    cm = np.arctan2(vxy[i, :, 1], vxy[i, :, 0])
    U = np.cos(cm)
    V = np.sin(cm)
    arrow.set_UVC(U, V)
    #scale = 1

    #arrow.set_UVC(scale*vxy[i, :, 0], scale*vxy[i, :, 1])
    arrow.set_offsets(data[i])

    return scat

def update_plot_old(i, data, scat, arrow):
    cm = np.arctan2(data[i, :, 1] - data[i - 1, :, 1], data[i, :, 0] - data[i - 1, :, 0])
    scat.set_offsets(data[i])
    scat.set_array(cm)
    U = np.cos(cm)
    V = np.sin(cm)
    arrow.set_UVC(U, V)
    arrow.set_offsets(data[i])

    return scat