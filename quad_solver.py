import numpy as np

def solver(curr_pos, delta_y):
    # a2 dx^2 + b dx - delta_y = 0
#    a1 = 0.04476
#    a2 = -1.869e-5
    a1 = 0.04655
    a2 = -1.9481e-5
    b = 2*a2*curr_pos+a1
    d2 = (b)**2 + 4*a2*delta_y
    sol2 = (-b + np.sqrt(d2))/(2*a2)
    return sol2

def loc_mapping(l):
    ## mapping: 400 - 1000 -> -0.5, 0.5
    ## delta_y changes from -8 ~ 8 to -1, 1
    ## use curr = 650 standard startpoint
    ## standard normalized pos: 0.132
    
    a1 = 0.04655
    a2 = -1.9481e-5
    b = 2*a2*650+a1
    return 0.1328 + ((l-650)**2*a2+b*(l-650))/8
    
    
    
# def loc_inv_mapping(l):
    
#     return 162.8 372.6 595.9