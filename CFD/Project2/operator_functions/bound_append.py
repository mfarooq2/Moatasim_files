import numpy as np
def bound_appender(Solution_Matrix,Nx, Ny,BCBt,BCLt,BCTt,BCRt):
    Final_matrix=np.zeros((Nx+2,Ny+2))
    Final_matrix[0,:]=BCLt
    Final_matrix[-1,1:Ny+1]=BCRt
    Final_matrix[1:Nx+1,0]=BCBt
    Final_matrix[:,-1]=BCTt
    
    Final_matrix[0,0]=(Final_matrix[0,1]+Final_matrix[1,0])/2
    Final_matrix[-1,0]=(Final_matrix[-2,0]+Final_matrix[-1,1])/2
    Final_matrix[-1,-1]=(Final_matrix[-2,-1]+Final_matrix[-1,-2])/2
    Final_matrix[0,-1]=(Final_matrix[0,-2]+Final_matrix[1,-1])/2

    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            Final_matrix[i,j]=Solution_Matrix[i-1,j-1]
    return Final_matrix

            