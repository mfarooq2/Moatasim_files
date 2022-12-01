import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from tqdm import tqdm
from multiprocessing import Pool, freeze_support
from func_list_vect import GenPointer,Grad,Div,BC_Div,Laplace,BC_Laplace,Adv,pointer_mapping,CG_solver,CG_solver_all,R_operator,S_operator,R_inv_operator,curl_operator,BC_Curl,inter_velocity

nx = 129
ny = 129
## Pointer and Grid
ip, iu, iv,idu = GenPointer(nx, ny)
resume=False
dx = 1 / nx
dy = 1 / ny

np_ = nx * ny
nu = 2*nx*ny - nx - ny
X,Y=np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny), indexing='ij')

###Boundaries
uBC_L = 0
uBC_R = 0
uBC_B = 0
uBC_T = 1
vBC_L = 0
vBC_R = 0
vBC_T = 0
vBC_B = 0
Re=400
v=1/Re
dt=0.9/((1/dx)+(1/dy))
dt=0.0035

p  = np.zeros((np_, 1))
u  = np.zeros((nu, 1))
qi = np.zeros((nu,1))
b  = np.zeros((nu,1))
bcL=BC_Laplace(uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,nu,iu,iv,nx,ny,dx,dy)
bcD = BC_Div(uBC_L, uBC_R, vBC_T, vBC_B,np_,ip,nx,ny,dx,dy)

cg_iter=1000
A_n=Adv(qi, uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,nu,iu,iv,nx,ny,dx,dy,dt,v)
Lu_n=Laplace(qi,nu,iu,iv,nx,ny,dx,dy,dt,v)+bcL

uf=qi + dt*(A_n+v*Lu_n)
D_rhs= (Div(uf,np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v)+bcD)/dt


args1=np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v
args2=np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v
#args3=np_,nu,nx,ny,dx,dy,iu,iv,ip
    
P=CG_solver_all([Grad,Div],D_rhs,p,args1,args2,args2,cg_iter)
u_new=uf-dt*Grad(P,np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v)

if resume==True:
    with open('/Users/moatasimfarooque/Documents/CFD/Moatasim_files/CFD/Project_3/test1.npy', 'rb') as f:
        qi = np.load(f)
        u_new = np.load(f)


## time stepping using fractial step
res=[]
res_10=[]
nt=200000
for it in tqdm(range(nt)):
    ## fractial step: stage 1
    A_n_1 = Adv(qi, uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,nu,iu,iv,nx,ny,dx,dy,dt,v)
    A_n= Adv(u_new, uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,nu,iu,iv,nx,ny,dx,dy,dt,v)
    RHS_b = S_operator(u_new,nu,iu,iv,nx,ny,dx,dy,dt,v) + dt*(3*(A_n)-(A_n_1))/2 +dt*v*bcL
    args=nu,iu,iv,nx,ny,dx,dy,dt,v
    u_int=np.random.rand(u_new.shape[0],u_new.shape[1])
    uf    = CG_solver(R_operator,RHS_b,u_new,args,cg_iter)

    ## fractial step: stage 2
    RHS_b = (Div(uf,np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v)+bcD)/dt
    
    args1=np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v
    args2=nu,iu,iv,nx,ny,dx,dy,dt,v
    args3=np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v
    
    
    pnew  = CG_solver_all([Grad,R_inv_operator,Div],RHS_b,P,args1,args2,args3,cg_iter)
    

    ## fractial step: stage 3 (assemble u_new)
    qi=u_new
    #P=np.random.rand(pnew.shape[0],pnew.shape[1])
    P=pnew
    u_new = uf-dt*R_inv_operator(Grad(pnew,np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v),nu,iu,iv,nx,ny,dx,dy,dt,v)
    norm_l2 = (norm(u_new))/(norm(qi))
    res.append(norm_l2)
    if it%10==0:
        
        print('iteration=',it)
        print('res=',norm_l2)
        
        with open('test1.npy', 'wb') as f:
            np.save(f, qi)
            np.save(f, u_new)
        
    if it%5==0:
        with open(f"snapshots/ulist_{it}.npy", 'wb') as f:
            np.save(f, u_new)

#pd.DataFrame({'u_new':u_new[:,0]}).to_csv('latest_data.csv')