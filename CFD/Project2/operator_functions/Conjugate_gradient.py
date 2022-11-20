import numpy as np

from operator_functions.Laplace_operator import LaplaceOpt_Aq,uBC_Laplace1,b_vector_creator
from operator_functions.AB_operator import A_operator,B_operator,uBC_Laplace
from numpy.linalg import norm
def CG_solver(Q,b,Nx,Ny,dx,dy,BCB,BCL,BCT,BCR,cg_iter):
    b_residual=[]
    b_vector=b_vector_creator(b,Nx,Ny,dx,dy,BCB,BCL,BCT,BCR)-uBC_Laplace1(Nx,Ny,dx,dy,BCB,BCL,BCT,BCR)
    d_old=b_vector-LaplaceOpt_Aq(Q,Nx,Ny,dx,dy)
    r_old=d_old


    for i in range(cg_iter):
        intermediate_vec=LaplaceOpt_Aq(d_old,Nx,Ny,dx,dy)
        alpha_factor=((r_old.T)@r_old)/((d_old.T)@(intermediate_vec))

        Q=Q+(alpha_factor[0,0])*d_old
        r_new=r_old-(alpha_factor[0,0])*(intermediate_vec)

        beta=((r_new.T)@(r_new))/((r_old.T)@(r_old))
        d_new=r_new+(beta[0,0])*d_old

        d_old=d_new
        r_old=r_new
        b_residual.append(norm(b_vector-LaplaceOpt_Aq(Q,Nx,Ny,dx,dy)))
    Solution_Matrix=np.zeros((Nx,Ny))
    k=0
    for i in range(Nx):
        for j in range(Ny):
            Solution_Matrix[i,j]=Q[k]
            k=k+1
            
    return Q,Solution_Matrix,b_residual

def CG_solver_transient(Q,kappa,Nx,Ny,dx,dy,BCB,BCL,BCT,BCR,dt,alpha):
    b=A_operator(Q,Nx,Ny,dx,dy,dt,alpha)+b_vector_creator(kappa,Nx,Ny,dx,dy,BCB,BCL,BCT,BCR)+uBC_Laplace(Nx,Ny,dx,dy,BCB,BCL,BCT,BCR,dt,alpha)
    d_old=b-B_operator(Q,Nx,Ny,dx,dy,dt,alpha)
    r_old=d_old
    b_residual=[]


    for i in range(100):
        intermediate_vec=B_operator(d_old,Nx,Ny,dx,dy,dt,alpha)
        
        alpha_factor=((r_old.T)@r_old)/((d_old.T)@(intermediate_vec))

        Q=Q+(alpha_factor[0,0])*d_old
        r_new=r_old-(alpha_factor[0,0])*(intermediate_vec)

        beta=((r_new.T)@(r_new))/((r_old.T)@(r_old))
        d_new=r_new+(beta[0,0])*d_old

        d_old=d_new
        r_old=r_new
        l2_norm=norm(b-B_operator(Q,Nx,Ny,dx,dy,dt,alpha))
        b_residual.append(l2_norm)
        if l2_norm<0.0001:
            break
    Solution_Matrix=np.zeros((Nx,Ny))
    k=0
    for i in range(Nx):
        for j in range(Ny):
            Solution_Matrix[i,j]=Q[k]
            k=k+1
            
    return Q,Solution_Matrix,b_residual

def CG_solver_transient_Matrix(matrix_B,Q,b,Nx,Ny,dx,dy,BCB,BCL,BCT,BCR,dt,alpha):
    d_old=b-B_operator(Q,Nx,Ny,dx,dy,dt,alpha)
    r_old=d_old

    for i in range(100):
        intermediate_vec=matrix_B@d_old
        
        alpha_factor=((r_old.T)@r_old)/((d_old.T)@(intermediate_vec))

        Q=Q+(alpha_factor[0,0])*d_old
        r_new=r_old-(alpha_factor[0,0])*(intermediate_vec)

        beta=((r_new.T)@(r_new))/((r_old.T)@(r_old))
        d_new=r_new+(beta[0,0])*d_old

        d_old=d_new
        r_old=r_new
    Solution_Matrix=np.zeros((Nx,Ny))
    k=0
    for i in range(Nx):
        for j in range(Ny):
            Solution_Matrix[i,j]=Q[k]
            k=k+1
            
    return Q,Solution_Matrix

