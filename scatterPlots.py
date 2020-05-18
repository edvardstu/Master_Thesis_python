import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import enum

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
    scat = plt.scatter(x_0, y_0, c=c, cmap='hsv')
    plt.quiver(x_0, y_0, np.cos(c), np.sin(c))

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    cbar = plt.colorbar(scat)
    cbar.ax.set_ylabel(r'$\theta$')
    cbar.set_clim(-2*np.pi, 2*np.pi)


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
        vxy[i] = np.stack((vx[i * offset].T, vy[i * offset].T)).T

    numframes = n_frames
    numpoints = n_particles

    x_0 = x[0]
    y_0 = y[0]
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
    scat = ax.scatter(x_0, y_0, c=c, cmap='jet')

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
                                  fargs=(xy, vxy, scat, arrow))
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #ani.save('test.mp4', writer=writer)
    #ani.save('test.gif', writer='imagemagick', fps=5)
    plt.show()


def update_plot(i, data, vxy, scat, arrow):
    #cm = np.arctan2(data[i, :, 1] - data[i - 1, :, 1], data[i, :, 0] - data[i - 1, :, 0])
    cm = np.arctan2(vxy[i, :, 1], vxy[i, :, 0])
    #cm = np.log(np.sqrt(vxy[i, :, 1]**2 + vxy[i, :, 0]**2))
    scat.set_offsets(data[i])
    scat.set_array(cm)
    #U = np.cos(cm)
    #V = np.sin(cm)
    #arrow.set_UVC(U, V)
    arrow.set_UVC(vxy[i, :, 0]*2, vxy[i, :, 1]*2)
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