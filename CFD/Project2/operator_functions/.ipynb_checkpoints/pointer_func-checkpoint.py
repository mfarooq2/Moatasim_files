import numpy as np
def pointer_vector(Nx, Ny):
    iQ=np.nan*np.zeros((Nx,Ny))
    ids=np.arange(start=0,stop=(Nx)*(Ny))
    k=0
    for i in range(Nx):
        for j in range(Ny):
            iQ[i,j]=int(ids[k])
            k=k+1
            if k==len(ids):
                break
        if k==len(ids):
            break
            
    return iQ.astype(int)