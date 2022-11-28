import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from tqdm import tqdm
from func_list import explicit_matrix,GenPointer,Grad,Div,BC_Div,Laplace,BC_Laplace,Adv,pointer_mapping,CG_solver,CG_solver_all,R_operator,S_operator,R_inv_operator,curl_operator,BC_Curl,inter_velocity
nx = 5
ny = 5
## Pointer and Grid
ip, iu, iv,idu = GenPointer(nx, ny)

dx = 1 / nx
dy = 1 / ny

np_ = nx * ny
nu = 2*nx*ny - nx - ny
dt=0.001
v=0.5
p  = np.zeros((np_, 1))
u  = np.zeros((nu, 1))

args=np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v
inp=np_
out=nu
Q=np.zeros((out,1))
A=explicit_matrix(Grad,out,inp,args)
#args=np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v
#A,i=explicit_matrix(Grad,out,inp,args)