import numpy as np

def fromRectangular(L,H, f_h):
    A = L*H
    H_new = np.sqrt(A*2/(1+f_h)*H/L)
    L_new = H_new*L/H
    h = H_new*f_h

    #print("For L: %.3f, and H: %.3f,\nto preserve the area a funnel with h/H = %.3f, should have the following dimensions:" %(L, H, f_h))
    #print("L: %.3f\nH: %.3f\nh: %.3f\n" %(L_new, H_new, h))
    print("%.3f\t%.3f\t%.3f\n" % (L_new, H_new, h))

#fromRectangular(25, 25, 18/20)

for i in range(0, 20):
    print("%d/20\t" % ((20-i)), end="")
    fromRectangular(25, 25, (20-i) / 20)