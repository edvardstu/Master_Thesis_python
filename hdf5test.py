import h5py
import numpy as np
import fileReader
from datetime import datetime

start1 = datetime.now()
f = h5py.File("/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/phaseDiagram/testHDF50.h5", "r")

#dset = f.get("dset").valuE
dset = np.array(f["dset"])
#print(dset)
#print(dset.shape)
#print(dset.dtype)

readTimeH5 = datetime.now()-start1

start2 = datetime.now()
fileName = "/home/edvardst/Documents/NTNU/Programming/Master_Thesis/Master_Thesis_c/results/periodic_2D/phaseDiagram/testHDF50.txt"
time, x, y, theta, vx, vy, n_particles, n_steps, D_r, deformation = fileReader.loadFileNew(fileName)
readTimeTXT = datetime.now()-start2

print("Readtime of HDF5: \t {}".format(readTimeH5))
print("Readtime of txt: \t {}".format(readTimeTXT))
print("Speedup, HDF5/txt: \t {}".format(readTimeTXT/readTimeH5))

