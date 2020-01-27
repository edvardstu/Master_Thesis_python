import numpy as np
from matplotlib import pyplot as plt
import integrationsSchemesTwoParticles as tp

def force(x1, x2):
    sigma_pp = 1.5
    r = abs(x2-x1)
    #f = 12 * ((2.0 * sigma_pp**2) / r**2 - 1.0) / r
    f =-5.0*(r - np.sqrt(2*sigma_pp**2))
    return f

def analytical(t):
    sigma_pp = 1.5
    d = np.sqrt(2*sigma_pp**2)
    v_0 = 10
    k = 5.0
    z_0 = 2
    return z_0*np.exp(-2*k*t) + (d-v_0/k)*(1-np.exp(-2*k*t))


steps = 15
dt = 0.05
sigma_pp=1.5
u_0 = 10.0



x1 = np.zeros(steps+1)
x2 = np.zeros(steps+1)
x1[0] = -1.0
x2[0] = 1.0

for i in range(1, steps+1):
    f1 = 0
    f2 = 0
    if abs(x1[i-1]-x2[i-1]) < np.sqrt(2*sigma_pp**2):
        f = force(x1[i-1], x2[i-1])
        f1 = -f
        f2 = f

    x1[i] = x1[i-1] + (u_0+f1)*dt
    x2[i] = x2[i-1] + (-u_0+f2)*dt


x1_AB = np.zeros(steps+1)
x2_AB = np.zeros(steps+1)
x1_AB[0] = -1.0
x2_AB[0] = 1.0
#x1_AB[1] = x1[1]
#x2_AB[1] = x2[1]

for i in range(1, steps+1):
    f = 0
    if i == 1:
        f = force(x1[i - 1], x2[i - 1])

    else:
        if abs(x1_AB[i-1]-x2_AB[i-1]) < 2*sigma_pp**2:
            if i==3:
                print(-force(x1_AB[i-1], x2_AB[i-1])+u_0)
            f += 3/2 *force(x1_AB[i-1], x2_AB[i-1])
        if abs(x1_AB[i-2]-x2_AB[i-2]) < 2*sigma_pp**2:
            if i ==3:
                print(-force(x1_AB[i-2], x2_AB[i-2])+u_0)
            f -= 1/2 *force(x1_AB[i-2], x2_AB[i-2])
    f1 = -f
    f2 = f
    #print((u_0+f1))
    x1_AB[i] = x1_AB[i-1] + (u_0+f1)*dt
    x2_AB[i] = x2_AB[i-1] + (-u_0+f2)*dt



time = np.linspace(0, steps*dt, num=steps+1)

plt.figure()

plt.plot(time, x2-x1, label='EM')
plt.plot(time, x2_AB-x1_AB, label='AB')
time2 = np.linspace(0, steps*dt, num=500)
plt.plot(time2, analytical(time2), label='Analytic')
plt.xlabel(r'$t$')
plt.ylabel(r'$\Delta x$')
plt.legend()
plt.savefig("AnalyticvsEMvAB.png", dpi=150)


x_EM_error = abs((x2-x1)-analytical(time))/analytical(time)
x_AB_error = abs((x2_AB-x1_AB)-analytical(time))/analytical(time)
plt.figure()
plt.plot(time, x_EM_error, label="EM")
plt.plot(time, x_AB_error, label="AB")
plt.xlabel(r'$t$')
plt.ylabel(r'$|\Delta x_a-\Delta x|/\Delta x_a$')
plt.legend()
plt.savefig("AnalyticvsEMvABError.png", dpi=150)


'''
plt.figure()
fileName = "Results/Integrators/EMSimple0.txt"
time_n, x_n, y_n, theta, vx, vy, n_particles, n_steps, D_r = tp.loadFile(fileName)
plt.plot(time, x1, label='New p1')
plt.plot(time, x2, label='New, p2')
plt.plot(time, x_n[:, 0], label='EM, p1')
plt.plot(time, x_n[:, 1], label='EM, p2')
time2 = np.linspace(0, steps*dt, num=500)
plt.plot(time2, analytical(time2)/2, label='particle 1, An')
plt.plot(time2, -analytical(time2)/2, label='particle 2, An')
'''


'''
plt.figure()
fileName = "Results/Integrators/ABSimple0.txt"
time_n, x_n, y_n, theta, vx, vy, n_particles, n_steps, D_r = tp.loadFile(fileName)
plt.plot(time, x1_AB, label='New p1')
plt.plot(time, x2_AB, label='New, p2')
plt.plot(time, x_n[:, 0], label='AB, p1')
plt.plot(time, x_n[:, 1], label='AB, p2')
time2 = np.linspace(0, steps*dt, num=500)
#plt.plot(time2, analytical(time2)/2, label='particle 1, An')
#plt.plot(time2, -analytical(time2)/2, label='particle 2, An')
'''

plt.show()


'''
time2 = np.linspace(0, steps*dt, num=500)
plt.plot(time2, analytical(time2)/2, label='particle 1, An')
plt.plot(time2, -analytical(time2)/2, label='particle 2, An')
'''



