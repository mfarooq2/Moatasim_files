import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from tqdm import tqdm
from func_list_vect import *
import json
v,args,cg_iter,nx,ny,dx,dy,np_,nu,X,Y,ip,iu,iv,idu,dt,p,u,qi,bcL,bcD,uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B=data_unloader(**json.load(open('/Users/moatasimfarooque/Documents/CFD/Moatasim_files/CFD/Project_3/data.json')))
v,args,cg_iter,nx,ny,dx,dy,np_,nu,X,Y,ip,iu,iv,idu,dt,p,u,qi,bcL,bcD,uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B=data_unloader(**json.load(open('/Users/moatasimfarooque/Documents/CFD/Moatasim_files/CFD/Project_3/data.json')))
Operator=[[Grad_Vec],[Laplace_Vec],[Div_Vec],[Adv_Vec]]
Operator_names=['Grad','Laplace','Div','Adv']
nu_list=[np_,nu,nu,nu]
#args=np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v
fig, axs = plt.subplots(2, 2)
k=0
for i in range(2):
    for j in range(2):
        s=explicit_matrix_combo(Operator[k],nu_list[k],args)
        axs[i, j].matshow(s)
        axs[i, j].set_title(f"{Operator_names[k]}")
        k=k+1
plt.show()