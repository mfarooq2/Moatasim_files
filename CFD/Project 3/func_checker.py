import numpy as np
import pandas as pd
from func_list import GenPointer,Grad,Div,BC_Div,Laplace,BC_Laplace,Adv,pointer_mapping,CG_solver
nx = 5
ny = 6
## Pointer and Grid
ip, iu, iv = GenPointer(nx, ny)

dx = 1 / nx
dy = 1 / ny

np_ = nx * ny
nu = 2*nx*ny - nx - ny
dt=0.01
v=1.6


###Boundaries
uBC_L = 0
uBC_R = 0
uBC_B = 0
uBC_T = 1
vBC_L = 0
vBC_R = 0
vBC_T = 0
vBC_B = 0

p  = np.zeros((np_, 1))
u  = np.zeros((nu, 1))
qi = np.zeros((nu,1))
b  = np.zeros((nu,1))
bcL=BC_Laplace(uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,nu,iu,iv,nx,ny,dx,dy)
bcD = BC_Div(uBC_L, uBC_R, vBC_T, vBC_B,np_,ip,nx,ny,dx,dy)
qi=np.zeros((nu,1))
q0=Adv(qi, uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,nu,iu,iv,nx,ny,dx,dy)
s=2