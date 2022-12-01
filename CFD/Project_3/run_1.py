import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from tqdm import tqdm
import json
from func_list_vect import *
#from func_list import GenPointer,Grad,Div,BC_Div,Laplace,BC_Laplace,Adv,pointer_mapping,CG_solver,CG_solver_all,R_operator,S_operator,R_inv_operator,curl_operator,BC_Curl,inter_velocity

v,args,cg_iter,nx,ny,dx,dy,np_,nu,X,Y,ip,iu,iv,idu,dt,p,u,qi,bcL,bcD,uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B=data_unloader(**json.load(open('/Users/moatasimfarooque/Documents/CFD/Moatasim_files/CFD/Project_3/data.json')))
A_n=Adv_Vec(qi, *args)
Lu_n=Laplace_Vec(qi,*args)+v*bcL

uf=qi + dt*(A_n+Lu_n)
D_rhs= (Div_Vec(uf,*args)+bcD)/dt
    
P=CG_solver_all([Grad_Vec,Div_Vec],D_rhs,p,args,cg_iter)
u_new=uf-dt*Grad_Vec(P,*args)
resume=0
if resume==True:
    with open('/Users/moatasimfarooque/Documents/CFD/Moatasim_files/CFD/Project_3/test1.npy', 'rb') as f:
        qi = np.load(f)
        u_new = np.load(f)


## time stepping using fractial step
res=[]
res_10=[]
nt=1000
dict_saver={}
dict_saver['qi']=[]
dict_saver['uf']=[]
dict_saver['u_new']=[]
dict_saver['P']=[]
for it in tqdm(range(nt)):
    ## fractial step: stage 1
    A_n_1 = Adv_Vec(qi, *args)
    A_n= Adv_Vec(u_new, *args)
    RHS_b = S_operator(u_new,*args) + dt*(3*(A_n)-(A_n_1))/2 +dt*v*bcL

    u_int=np.random.rand(u_new.shape[0],u_new.shape[1])
    uf    =CG_solver(R_operator,RHS_b,u_new,args,500)

    ## fractial step: stage 2
    RHS_b = (Div_Vec(uf,*args)+bcD)/dt
    
    
    
    pnew  = CG_solver_all([Grad_Vec,R_inv_operator,Div_Vec],RHS_b,P,args,cg_iter)
    

    ## fractial step: stage 3 (assemble u_new)
    qi=u_new
    #P=np.random.rand(pnew.shape[0],pnew.shape[1])
    
    u_new = uf-dt*R_inv_operator(Grad_Vec(pnew,*args),*args)
    #norm_l2 = (norm(u_new-qi))/(norm(qi))
    norm_l2 = 100*abs(norm(u_new)-norm(qi))/norm(qi)
    norm_P = 100*abs(norm(pnew-P))/norm(P)
    P=pnew
    res.append(norm_l2)
    dict_saver['qi'].append(norm(qi))
    dict_saver['uf'].append(norm(uf))
    dict_saver['u_new'].append(norm(u_new))
    dict_saver['P'].append(norm(P))
    
    if it%10==0:
        print('time=',it*dt)
        print('res=',norm_l2,'%')
        print('res_P=',norm_P,'%')
        
        with open('test_2.npy', 'wb') as f:
            np.save(f, qi)
            np.save(f, uf)
            np.save(f, u_new)
            np.save(f, P)
        
    #if it%25==0:
        #u_vec=norm(u_new[0:iu[-1,-1]+1])
        #v_vec=norm(u_new[iu[-1,-1]+1:])
        #print('CFL_x=',u_vec*((dt/dx)))
        #print('CFL_y=',v_vec*((dt/dy)))
        #dt=0.95*dt
        #with open(f"snapshots/ulist_{it}.npy", 'wb') as f:
        #    np.save(f, u_new)
pd.DataFrame(dict_saver).to_csv('dict_saver.csv',index=False)