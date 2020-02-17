import numpy as np
from matplotlib import pyplot as plt
import fileReader
from scipy import stats

fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/infWell/test0.txt"

time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)

L = 50.0
v_0 = 10
v_s = 0.5
D_r = 5.0


padding = 1.0
y_max = np.amax(y)

#Plot system
plt.scatter(x, y, s=0.1)

plt.plot([-0.0, L],[0.0, 0.0], "black")
plt.plot([-0.0, 0.0],[0.0, y_max], "black")
plt.plot([L, L],[0.0, y_max], "black")

plt.axis('scaled')
plt.xlim(-padding, L+padding)
plt.ylim(-padding, y_max + padding)
plt.xlabel(r"x")
plt.ylabel(r"z")
plt.tight_layout()
plt.savefig("plots/infWell/positionSampling.png", dpi=200)

#Create histogram
n_bins = 30#25
plt.figure()
newArray = plt.hist(y, bins=n_bins, density=True)
plt.xlabel(r'$z$')
plt.ylabel(r'Normalized position sampling')
plt.tight_layout()
plt.savefig("plots/infWell/histogram.png", dpi=200)





plt.figure()

#Historgram
yNewValues = (newArray[1][1:n_bins+1] + newArray[1][0:n_bins])/2
plt.plot(yNewValues, np.log(newArray[0]), label="Histogram")

#Regression og linspace
slope, intercept, r_value, p_value, std_err = stats.linregress(yNewValues, np.log(newArray[0]))
y_ls = np.linspace(yNewValues.min(), yNewValues.max(), 10)

#Analytic value
lambda_e = v_0**2/(2*D_r*v_s) * (1-7/4*(v_s/v_0)**2)
plt.plot(y_ls, y_ls*(-1/lambda_e)+intercept, label="Analytic")

#Linear regression
plt.plot(y_ls, y_ls*slope+intercept, label="Lin. Reg.")

#MLE
lambda_MLE = len(y)/np.sum(y)
plt.plot(y_ls, (-lambda_MLE*y_ls)+intercept, label="MLE")


plt.xlabel(r"$z$")
plt.ylabel(r"Logarithm of probability density")
plt.legend()
plt.savefig("plots/infWell/gradient.png", dpi=200)

print("Standard error, lin. reg.: %f" % (std_err))
print("Standard error, MLE:       %f" % (lambda_MLE/np.sqrt(len(y))))


plt.show()

#Weigh the linear regression more in the beginning. According to the number of values in each bin.