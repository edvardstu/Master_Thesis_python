import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def plotState(x, y, vx, vy, i, R):
    plt.figure()
    x_0 = x[i]
    y_0 = y[i]
    v_0 = 10
    #c = np.arctan2(y[i + 1] - y[i], x[i + 1] - x[i])
    c = np.arctan2(vy[i], vx[i])
    # c= theta[i]
    scat = plt.scatter(x_0, y_0, c=c, cmap='hsv')
    plt.quiver(x_0, y_0, np.cos(c), np.sin(c))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    cbar = plt.colorbar(scat)
    cbar.ax.set_ylabel(r'$\theta$')
    cbar.set_clim(-2*np.pi, 2*np.pi)


    # Add confining circle
    theta_circle = np.linspace(0, 2 * np.pi, 1500)
    plt.plot(R * np.cos(theta_circle), R * np.sin(theta_circle), color=(0, 0, 0, 0.5))

    '''
    beta1 = np.arctan2(y[i], x[i])
    beta2 = np.arctan2(y[i-1], x[i-1])
    beta = (beta1 + beta2)/2
    p_phi = vx[i]/v_0*np.sin(-beta) + vy[i]/v_0*np.cos(-beta)
    Pi = np.mean(p_phi)
    print(Pi)

    print(np.mean(np.sqrt(vx[i]**2+vy[i]**2)))
    '''

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


def run_animation(x, y, theta, R, n_steps, n_particles, n_frames):
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


def update_plot(i, data, scat, arrow):
    cm = np.arctan2(data[i, :, 1] - data[i - 1, :, 1], data[i, :, 0] - data[i - 1, :, 0])
    scat.set_offsets(data[i])
    scat.set_array(cm)
    U = np.cos(cm)
    V = np.sin(cm)
    arrow.set_UVC(U, V)
    arrow.set_offsets(data[i])

    return scat