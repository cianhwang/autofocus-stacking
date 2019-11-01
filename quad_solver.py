import numpy as np

def solver(curr_pos, delta_y):
#    a1 = 0.04476
#    a2 = -1.869e-5
    a1 = 0.04655
    a2 = -1.9481e-5
    b = 2*a2*curr_pos+a1
    d2 = (b)**2 + 4*a2*delta_y
    sol2 = (-b + np.sqrt(d2))/(2*a2)
    return sol2
