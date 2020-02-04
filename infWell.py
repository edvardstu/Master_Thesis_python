import numpy as np
from matplotlib import pyplot as plt
import fileReader
from scipy import stats

fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/infWell/test0.txt"

time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)

L = 5.0
v_0 = 10
v_s = 1.0
D_r = 10.0


padding = 1.0
h_factor = 7

plt.plot(x, y)

plt.plot([-0.0, L],[0.0, 0.0], "black")
plt.plot([-0.0, 0.0],[0.0, h_factor*L], "black")
plt.plot([L, L],[0.0, h_factor*L], "black")

plt.axis('scaled')
plt.xlim(-padding, L+padding)
plt.ylim(-padding, h_factor*L + padding)

n_bins = 25
plt.figure()
newArray = plt.hist(y, bins=n_bins, density=True)
plt.xlabel(r'$y$')
plt.ylabel(r'Position count')


plt.figure()

yNewValues = (newArray[1][1:n_bins+1] + newArray[1][0:n_bins])/2
plt.plot(yNewValues, np.log(newArray[0]), label="Histogram")
slope, intercept, r_value, p_value, std_err = stats.linregress(yNewValues, np.log(newArray[0]))
y_ls = np.linspace(yNewValues.min(), yNewValues.max(), 10)
plt.plot(y_ls, y_ls*slope+intercept, label="Regression")

lambda_e = v_0**2/(2*D_r*v_s) * (1-7/4*(v_s/v_0)**2)
plt.plot(y_ls, y_ls*(-1/lambda_e)+intercept, label="Analytic")
plt.legend()
plt.show()

#Weigh the linear regression more in the beginning. According to the number of values in each bin.