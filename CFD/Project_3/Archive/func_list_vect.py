import numpy as np
import pandas as pd
from multiprocessing import Pool, freeze_support
from numpy.linalg import norm
from numba import njit
import multiprocessing
from tqdm import tqdm
def GenPointer(nx, ny):
    ## Memory allocation
    ip = np.nan*np.ones((nx,ny))
    iu = np.nan*np.ones((nx,ny))
    iv = np.nan*np.ones((nx,ny))
    idu = np.nan*np.ones((nx,ny))

    ## Pointer matrix for P
    id_p = 0 # index to be used in vector variable P
    for i in range(0,nx):
        for j in range(0,ny):
            ip[i, j] = id_p
            id_p = id_p + 1 
        ## Pointer matrix for P
    
    id_uni = 0 # index to be used for universal_calculation
    for i in range(0,nx):
        for j in range(0,ny):
            idu[i, j] = id_uni
            id_uni = id_uni + 1 

    ## Pointer matrix for ux
    id_u = 0  # index to be used in vector variable u = [ux; uy]
    for i in range(1,nx):
        for j in range(0,ny):
            iu[i, j] = id_u
            id_u = id_u + 1

    ## Pointer matrix for uy
    for i in range(0,nx):
        for j in range(1,ny):
            iv[i, j] = id_u
            id_u = id_u + 1
    ip[np.isnan(ip)]=0
    iv[np.isnan(iv)]=0
    iu[np.isnan(iu)]=0
    idu[np.isnan(idu)]=0
    return ip.astype(int),iu.astype(int),iv.astype(int),idu.astype(int)
 
def Grad(qi,np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v):
    ## Gradient operator: 
    #       input: p-type (np elements)
    #       output: u-type (nu elements)
    ## Input size check:
    qo = np.nan*np.ones((nu,1))

    ## inner domain
    ## x-direction gradient

    qo[iu[1:nx, 0:ny]] = ( -qi[ip[0:nx-1, 0:ny]] + qi[ip[1:nx, 0:ny]] ) / dx

    ## y-direction gradient

    qo[iv[0:nx, 1:ny]] = ( -qi[ip[0:nx, 0:ny-1]] + qi[ip[0:nx, 1:ny]] ) / dy
    return qo
 
def Div(qi,np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v):

    ## Initialize output
    qo = np.nan*np.ones((np_,1))

    ## inner domain

    qo[ip[1:nx-1, 1:ny-1]] = ((- qi[iu[1:nx-1, 1:ny-1]] + qi[iu[2:nx, 1:ny-1]] ) / dx) + ((- qi[iv[1:nx-1, 1:ny-1]] + qi[iv[1:nx-1, 2:ny]] ) / dy)
    ## Edges
    ## bottom inner
    j=0
    for i in range(1,nx-1):
        qo[ip[1:nx-1, 0]] = ((- qi[iu[1:nx-1, 0]] + qi[iu[2:nx, 0]] ) / dx) + (( + qi[iv[1:nx-1, 0+1]] ) / dy)  ## -qi[iv[i, j]]

    ## top inner
    j = -1
    for i in range(1,nx-1):
        qo[ip[i, j]] = (( - qi[iu[i, j]] + qi[iu[i+1, j]] ) / dx) + (( - qi[iv[i, j]]) / dy)   ## + qi[iv[i, j+1]]

    ## left inner
    i = 0
    for j in range(1,ny-1):
        qo[ip[i, j]] = ((+ qi[iu[i+1, j]] ) / dx)  + (( - qi[iv[i, j]] + qi[iv[i, j+1]] ) / dy) ## - qi[iu[i, j]] 

    ## right inner
    i=-1
    for j in range(1,ny-1):
        qo[ip[i, j]] = ((- qi[iu[i, j]] ) / dx) + ((- qi[iv[i, j]] + qi[iv[i, j+1]] ) / dy) ##+ qi[iu[i+1, j]] 

    ## Corners
    ## bottom left (pinning)
    i = 0
    j = 0
    #qo[ip[i, j]] = (( + qi[iu[i+1, j]] ) / dx) + (( + qi[iv[i, j+1]] ) / dy)
    qo[ip[i, j]] = 0

    # bottom right (pinning)
    i=-1
    j=0
    #qo[ip[i, j]] =((- qi[iu[i, j]] ) / dx) + ( + qi[iv[i, j+1]] ) / dy)
    qo[ip[i, j]] =0

    ## top left
    i = 0
    j = -1
    qo[ip[i, j]] = ((+ qi[iu[i+1, j]] ) / dx) + (( - qi[iv[i, j]]) / dy) ## - qi[iu[i, j]]   ## + qi[iv[i, j+1]] 

    ## top right
    i=-1
    j=-1
    qo[ip[i, j]] =((- qi[iu[i, j]] ) / dx) + ((- qi[iv[i, j]] ) / dy) # ## + qi[iu[i+1, j]]  + qi[iv[i, j+1]] 
    
    return qo

def BC_Div(uBC_L, uBC_R, vBC_T, vBC_B,np_,ip,nx,ny,dx,dy):
    
    ## BC vector for divergence operator: 
    #       input: BCs
    #       output: p-type (np elements)

    ## Initialize output
    bcD = np.zeros((np_, 1))

    ## Edges
    # bottom inner
    i=-1
    for j in range(1,ny-1):
        bcD[ip[i, j]] =   -vBC_B / dy 

    # top inner
    j = -1
    for i in range(1,nx-1):
        bcD[ip[i, j]] =   vBC_T / dy   # qi(iv(i, j+1))
    # left inner
    i = 0
    for j in range(1,ny-1):
        bcD[ip[i, j]] = - uBC_L / dx

    # right inner
    i=-1
    for j in range(1,ny-1):
        bcD[ip[i, j]] = uBC_R / dx


    ## Corners
    # bottom left (need pinning)
    i = 0
    j = 0
    bcD[ip[i, j]] = 0

    # bottom right
    i=-1
    j=0
    bcD[ip[i, j]] = 0
    

    # top left
    i = 0
    j = -1
    bcD[ip[i, j]] = (- uBC_L / dx) + (vBC_T / dy) # - qi[iu[i, j]] + qi(iv(i, j+1))

    # top right
    i=-1
    j=-1
    bcD[ip[i, j]] = (+ uBC_R / dx) + (vBC_T / dy) # ## + qi[iu[i+1, j]]  + qi[iv[i, j+1]] 

    return bcD
 
def Laplace(qi,nu,iu,iv,nx,ny,dx,dy,dt,v ):
    
    ## Laplace operator: 
    #       input: u-type (nu elements)
    #       output: u-type (nu elements)

    ## Initialize output
    qo=np.nan*np.ones((nu,1))

    ## 1. ex-Component
    ## inner domain

    qo[iu[2:nx-1, 1:ny-1]] = (+qi[iu[1:nx-2, 1:ny-1]] -2*qi[iu[2:nx-1, 1:ny-1]] + qi[iu[3:nx, 1:ny-1]] ) / (dx**2) + ( +qi[iu[2:nx-1, 0:ny-2]] -2*qi[iu[2:nx-1, 1:ny-1]] + qi[iu[2:nx-1, 2:ny]] ) / (dy**2)

    #qo[iu[i, j]] = (+qi[iu[i-1, j]] -2*qi[iu[i, j]] + qi[iu[i+1, j]] ) / (dx**2) + ( +qi[iu[i, j-1]] -2*qi[iu[i, j]] + qi[iu[i, j+1]] ) / (dy**2) qo[iu[2:nx-1, 1:ny-1]]

    ## Edges
    # left inner 
    i = 1
    for j in range(1,ny-1):
            qo[iu[i, j]] = (-2*qi[iu[i, j]] + qi[iu[i+1, j]] ) / (dx**2) + ( +qi[iu[i, j-1]] -2*qi[iu[i, j]] + qi[iu[i, j+1]] ) / (dy**2) # + uBC_L / (dx^2)

    # bottom inner
    j = 0
    for i in range(2,nx-1):
            qo[iu[i, j]] =   ( +qi[iu[i-1, j]] -2*qi[iu[i, j]] + qi[iu[i+1, j]] ) / (dx**2) + ( -qi[iu[i, j]]   -2*qi[iu[i, j]] + qi[iu[i, j+1]] ) / (dy**2) #   ## + 2*uBC_B / (dy^2) 

    #right inner
    i=-1
    for j in range(1,ny-1):
        qo[iu[i, j]] = (+qi[iu[i-1, j]] -2*qi[iu[i, j]] ) / (dx**2) + ( +qi[iu[i, j-1]] -2*qi[iu[i, j]] + qi[iu[i, j+1]] ) / (dy**2) # qi[iu[i+1, j]] 
    

    #top inner
    j=-1
    for i in range(2,nx-1):
        qo[iu[i, j]] = (+qi[iu[i-1, j]] -2*qi[iu[i, j]] + qi[iu[i+1, j]] ) / (dx**2) + ( +qi[iu[i, j-1]] -2*qi[iu[i, j]]  ) / (dy**2)  #+ qi[iu[i, j+1]]
        

    ## Corners
    # bottom left
    i = 1 
    j = 0
    qo[iu[i, j]] =(-2*qi[iu[i, j]] + qi[iu[i+1, j]] ) / (dx**2) + ( -qi[iu[i, j]]   -2*qi[iu[i, j]] + qi[iu[i, j+1]] ) / (dy**2) # + uBC_L   / (dx^2)   # + 2*uBC_B / (dy^2) 

    # bottom right
    i=-1
    j=0    
    qo[iu[i, j]] = (+qi[iu[i-1, j]] -2*qi[iu[i, j]]  ) / (dx**2) + (-2*qi[iu[i, j]] + qi[iu[i, j+1]] ) / (dy**2)  #+ qi[iu[i+1, j]] +qi[iu[i, j-1]]

    # top left
    i=1
    j=-1
    qo[iu[i, j]] = ( -2*qi[iu[i, j]] + qi[iu[i+1, j]] ) / (dx**2) + ( +qi[iu[i, j-1]] -2*qi[iu[i, j]]  ) / (dy**2)  ##+qi[iu[i-1, j]] + qi[iu[i, j+1]]
    

    # top right
    i=-1
    j=-1
    qo[iu[i, j]] = (+qi[iu[i-1, j]] -2*qi[iu[i, j]]  ) / (dx**2) + ( +qi[iu[i, j-1]] -2*qi[iu[i, j]]  ) / (dy**2) #+ qi[iu[i+1, j]] + qi[iu[i, j+1]]
    


    ## 2. ey-Component
    ## inner domain

    qo[iv[1:nx-1, 2:ny-1]] = (+qi[iv[0:nx-2, 2:ny-1]] -2*qi[iv[1:nx-1, 2:ny-1]] + qi[iv[2:nx, 2:ny-1]] ) / (dx**2) + ( +qi[iv[1:nx-1, 1:ny-2]] -2*qi[iv[1:nx-1, 2:ny-1]] + qi[iv[1:nx-1, 3:ny]] ) / (dy**2)


    ## Edges
    # left inner 
    i = 0
    for j in range(2,ny-1):
        qo[iv[i, j]] = (-qi[iv[i, j]] -2*qi[iv[i, j]] + qi[iv[i+1, j]] ) / (dx**2) + ( +qi[iv[i, j-1]] -2*qi[iv[i, j]] + qi[iv[i, j+1]] ) / (dy**2) # + 2uBC_L / (dx^2)

    # bottom inner
    j = 1
    for i in range(1,nx-1):
            qo[iv[i, j]] =   ( +qi[iv[i-1, j]] -2*qi[iv[i, j]] + qi[iv[i+1, j]] ) / (dx**2) + (    -2*qi[iv[i, j]] + qi[iv[i, j+1]] ) / (dy**2) #   ## + uBC_B / (dy^2) 

    #right inner
    i=-1
    for j in range(2,ny-1):
        qo[iv[i, j]] = (+qi[iv[i-1, j]] -2*qi[iv[i, j]] - qi[iv[i, j]]) / (dx**2) + ( +qi[iv[i, j-1]] -2*qi[iv[i, j]] + qi[iv[i, j+1]] ) / (dy**2) # 2UBCR/dx**2
    

    #top inner
    j=-1
    for i in range(1,nx-1):
        qo[iv[i, j]] = (+qi[iv[i-1, j]] -2*qi[iv[i, j]] + qi[iv[i+1, j]] ) / (dx**2) + ( +qi[iv[i, j-1]] -2*qi[iv[i, j]]  ) / (dy**2)  #+ ubC_t/(dy**2)


    ## Corners
    # bottom left
    i = 0 
    j = 1
    qo[iv[i, j]] =(-qi[iv[i, j]] -2*qi[iv[i, j]] + qi[iv[i+1, j]] ) / (dx**2) + (    -2*qi[iv[i, j]] + qi[iv[i, j+1]] ) / (dy**2) # + uBC_L   / (dx^2)   # 2uBC_L / (dx^2) + uBC_B / (dy^2) 

    # bottom right
    i=-1
    j=1    
    qo[iv[i, j]] = (+qi[iv[i-1, j]] -2*qi[iv[i, j]] - qi[iv[i, j]]) / (dx**2) +  (    -2*qi[iv[i, j]] + qi[iv[i, j+1]] ) / (dy**2)  # 2UBCR/dx**2 + uBC_B / (dy^2) 

    # top left
    i=0
    j=-1
    qo[iv[i, j]] = (-qi[iv[i, j]] -2*qi[iv[i, j]] + qi[iv[i+1, j]] ) / (dx**2) + ( +qi[iv[i, j-1]] -2*qi[iv[i, j]]  ) / (dy**2)   ##+ 2uBC_L / (dx^2) + ubC_t/(dy**2)
    

    # top right
    i=-1
    j=-1
    qo[iv[i, j]] = (+qi[iv[i-1, j]] -2*qi[iv[i, j]] - qi[iv[i, j]]) / (dx**2) +( +qi[iv[i, j-1]] -2*qi[iv[i, j]]  ) / (dy**2)  # 2UBCR/dx**2 + ubC_t/(dy**2)

    return qo

def BC_Laplace(uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,nu,iu,iv,nx,ny,dx,dy):
    
    

 
    ## BC vector for divergence operator: 
    #       input: BCs
    #       output: u-type (nu elements)

    ## Input size check:
    # input BC's are all scalars


    ## Initialize output
    bcL =np.zeros((nu,1)) ;

    ## 1. U-Component
    ## inner domain


    ## Edges
    # left inner 
    i = 1 
    for j in range(1,ny-1):
        bcL[iu[i, j]] = +   uBC_L / (dx**2)
    ## bottom inner
    j = 0
    for i in range(2,nx-1):
        bcL[iu[i, j]] = + 2*uBC_B / (dy**2)
    ## right inner
    i = -1
    for j in range(1,ny-1):
         bcL[iu[i, j]] = +   uBC_R / (dx**2)


    # top inner
    j=-1
    for i in range(2,nx-1):
         bcL[iu[i, j]] = +   uBC_T / (dy**2)


    ## Corners
    # bottom left
    i = 1
    j = 0
    bcL[iu[i, j]] = + uBC_L / (dx**2) + 2*uBC_B / (dy**2)  

    # bottom right
    i=-1
    j=0
    bcL[iu[i, j]] =  uBC_R / (dx**2) + 2*uBC_B / (dy**2)

    # top left
    i=1
    j=-1
    bcL[iu[i, j]] =  uBC_L / (dx**2) +  uBC_T / (dy**2)


    ## top right
    i=-1
    j=-1
    bcL[iu[i, j]] =  uBC_R / (dx**2) +  uBC_T / (dy**2)


    #### 2. V-Component
    ## inner domain


    ## Edges
    # left inner 
    i = 0 
    for j in range(2,ny-1):
        bcL[iv[i, j]] = + 2*vBC_L / (dx**2)
    ## bottom inner
    j = 1
    for i in range(1,nx-1):
        bcL[iv[i, j]] = + vBC_B / (dy**2) 
    ## right inner
    i = -1
    for j in range(2,ny-1):
         bcL[iv[i, j]] = 2*vBC_R/dx**2


    # top inner
    j=-1
    for i in range(1,nx-1):
         bcL[iv[i, j]] = +   vBC_T / (dy**2)


    ## Corners
    # bottom left
    i = 1
    j = 0
    bcL[iv[i, j]] = + 2*vBC_L / (dx**2) + vBC_B / (dy**2) 

    # bottom right
    i=-1
    j=0
    bcL[iv[i, j]] =  2*vBC_R/dx**2 +  vBC_B / (dy**2) 

    # top left
    i=1
    j=-1
    bcL[iv[i, j]] = 2*vBC_L / (dx**2) +   vBC_T / (dy**2)


    ## top right
    i=-1
    j=-1
    bcL[iv[i, j]] =  2*vBC_R/dx**2+   vBC_T / (dy**2)

    return bcL
 
def Adv(qi, uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,nu,iu,iv,nx,ny,dx,dy,dt,v):
    
    
    ## advection operator (BC embedded): -\nabla \cdot (uu) 
    ##      input: u-type (nu elements)
    ##       output: u-type (nu elements)
    #


    ## Initialize output
    qo = np.nan*np.ones((nu,1))

    ## 1. U-Component
    ## inner domain

    qo[iu[2:nx-1, 1:ny-1]] = - (1/dx) * ( - ( qi[iu[1:nx-2,1:ny-1]  ] + qi[iu[2:nx-1,1:ny-1]] ) / 2 * ( qi[iu[1:nx-2,1:ny-1  ]] + qi[iu[2:nx-1,1:ny-1  ]] ) / 2   \
                                            + ( qi[iu[2:nx-1,1:ny-1  ]] + qi[iu[3:nx,1:ny-1  ]] ) / 2 * ( qi[iu[2:nx-1,1:ny-1  ]] + qi[iu[3:nx,1:ny-1  ]] ) / 2 )   \
                               - (1/dy) * ( - ( qi[iu[2:nx-1,0:ny-2]] + qi[iu[2:nx-1,1:ny-1  ]] ) / 2 * ( qi[iv[1:nx-2,1:ny-1  ]] + qi[iv[2:nx-1,1:ny-1  ]] ) / 2          \
                                            + ( qi[iu[2:nx-1,1:ny-1  ]] + qi[iu[2:nx-1,2:ny]] ) / 2 * ( qi[iv[1:nx-2,2:ny]] + qi[iv[2:nx-1,2:ny]] ) / 2 )                 

    ## Edges
    ## left inner 

    qo[iu[1,1:ny-1]] = - ( - ( uBC_L           + qi[iu[1 ,1:ny-1  ]] ) / 2 * ( uBC_L           + qi[iu[1  ,1:ny-1  ]] ) / 2          \
                               + ( qi[iu[1  ,1:ny-1  ]] + qi[iu[1+1,1:ny-1  ]] ) / 2 * ( qi[iu[1  ,1:ny-1  ]] + qi[iu[1+1,1:ny-1  ]] ) / 2 ) / dx  \
                           - ( - ( qi[iu[1  ,0:ny-2]] + qi[iu[1  ,1:ny-1  ]] ) / 2 * ( qi[iv[1-1,1:ny-1  ]] + qi[iv[1  ,1:ny-1  ]] ) / 2          \
                               + ( qi[iu[1  ,1:ny-1  ]] + qi[iu[1  ,2:ny]] ) / 2 * ( qi[iv[1-1,2:ny]] + qi[iv[1  ,2:ny]] ) / 2 ) / dy                

    # bottom inner
    i=np.nan
    qo[iu[2:nx-1, 0]] = - ( - ( qi[iu[1:nx-2,0  ]] + qi[iu[2:nx-1  ,0  ]] ) / 2 * ( qi[iu[1:nx-2,0  ]] + qi[iu[2:nx-1  ,0  ]] ) / 2          \
                               + ( qi[iu[2:nx-1  ,0  ]] + qi[iu[3:nx,0  ]] ) / 2 * ( qi[iu[2:nx-1  ,0  ]] + qi[iu[3:nx,0  ]] ) / 2 ) / dx   \
                           - ( -   uBC_B                                   *   vBC_B                                            \
                               + ( qi[iu[2:nx-1  ,0  ]] + qi[iu[2:nx-1  ,0+1]] ) / 2 * ( qi[iv[1:nx-2,0+1]] + qi[iv[2:nx-1  ,0+1]] ) / 2 ) / dy      

    # right inner
    i=-1
    j=np.nan

    qo[iu[i, 1:ny-1]] = - (1/dx) * ( - ( qi[iu[i-1,1:ny-1  ]] + qi[iu[i  ,1:ny-1  ]] ) / 2 * ( qi[iu[i-1,1:ny-1  ]] + qi[iu[i  ,1:ny-1 ]] ) / 2         \
                                            + ( qi[iu[i  ,1:ny-1  ]] + qi[iu[i+1,1:ny-1  ]] ) / 2 * ( qi[iu[i  ,1:ny-1  ]] + qi[iu[i+1,1:ny-1  ]] ) / 2 )   \
                               - (1/dy) * ( - ( qi[iu[i  ,0:ny-2]] +uBC_R ) / 2 * ( qi[iv[i-1,1:ny-1  ]] + uBC_R ) / 2          \
                                            + ( qi[iu[i  ,1:ny-1  ]] + qi[iu[i  ,2:ny]] ) / 2 * ( qi[iv[i-1,2:ny]] + qi[iv[i  ,2:ny]] ) / 2 )      

    # top inner
    j=-1
    qo[iu[2:nx-1, j]] = - (1/dx) * ( - ( qi[iu[1:nx-2,j  ]] + qi[iu[2:nx-1  ,j  ]] ) / 2 * ( qi[iu[1:nx-2,j  ]] + qi[iu[2:nx-1  ,j  ]] ) / 2         \
                                            + ( qi[iu[2:nx-1  ,j  ]] + qi[iu[3:nx,j  ]] ) / 2 * ( qi[iu[2:nx-1  ,j  ]] + qi[iu[3:nx,j  ]] ) / 2 )   \
                               - (1/dy) * ( - ( qi[iu[2:nx-1  ,j-1]] + qi[iu[2:nx-1  ,j  ]] ) / 2 * ( qi[iv[1:nx-2,j  ]] + qi[iv[2:nx-1  ,j  ]] ) / 2          \
                                            + ( uBC_T*vBC_T ) )

    ## Corners
    # bottom left
    i = 1
    j = 0
    qo[iu[i, j]] = - (1/dx) * ( - ( uBC_L           + qi[iu[i  ,j  ]] ) / 2 * ( uBC_L           + qi[iu[i  ,j  ]] ) / 2          \
                                            + ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 )   \
                           - ( -   uBC_B                                   *   vBC_B                                            \
                               + ( qi[iu[i  ,j  ]] + qi[iu[i  ,j+1]] ) / 2 * ( qi[iv[i-1,j+1]] + qi[iv[i  ,j+1]] ) / 2 ) / dy  

    #bottom right
    i=-1
    j=0
    qo[iu[i, j]] =  - (1/dx) * ( - ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2         \
                                            + ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 )   \
                           - ( -   uBC_B                                   *   vBC_B                                            \
                               + ( qi[iu[i  ,j  ]] + qi[iu[i  ,j+1]] ) / 2 * ( qi[iv[i-1,j+1]] + qi[iv[i  ,j+1]] ) / 2 ) / dy  


    # top left
    i= 1
    j=-1
    qo[iu[i, j]] =  - (1/dx) * ( - ( uBC_L           + qi[iu[i  ,j  ]] ) / 2 * ( uBC_L           + qi[iu[i  ,j  ]] ) / 2          \
                                            + ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 )   \
                              - (1/dy) * ( - ( qi[iu[i  ,j-1]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
                                            + ( uBC_T*vBC_T ) )     


    # top right
    i=-1
    j=-1
    qo[iu[i, j]] =  - (1/dx) * ( - ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2         \
                                            + ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 )   \
                                               - (1/dy) * ( - ( qi[iu[i  ,j-1]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
                                            + ( uBC_T*vBC_T ) )  



    ## 2. V-Component
    ## inner domain
    i=np.nan
    j=np.nan
    qo[iv[1:nx-1, 2:ny-1]] = - (1/dx) * ( - ( qi[iu[0:nx-2,2:ny-1  ]] + qi[iu[1:nx-1,2:ny-1  ]] ) / 2 * ( qi[iv[0:nx-2,2:ny-1  ]] + qi[iv[1:nx-1  ,2:ny-1  ]] ) / 2          \
                                    + ( qi[iu[2:nx ,1:ny-2  ]] + qi[iu[2:nx,2:ny-1  ]] ) / 2 * ( qi[iv[2:nx  ,2:ny-1  ]] + qi[iv[1:nx-1,2:ny-1 ]] ) / 2 )   \
                       - (1/dy) * ( - ( qi[iv[1:nx-1  ,1:ny-2]] + qi[iv[1:nx-1,2:ny-1  ]] ) / 2 * ( qi[iv[1:nx-1,1:ny-2  ]] + qi[iv[1:nx-1  ,2:ny-1  ]] ) / 2          \
                                    + ( qi[iv[1:nx-1  ,3:ny  ]] + qi[iv[1:nx-1  ,2:ny-1]] ) / 2 * ( qi[iv[1:nx-1,3:ny]] + qi[iv[1:nx-1  ,2:ny-1]] ) / 2 ) 

    ## Edges
    # left inner
    i=0
    j=np.nan
    qo[iv[i, 2:ny-1]] = - (1/dx) * ( - ( uBC_L*vBC_L)         \
                                + ( qi[iu[i+1 ,2:ny-1  ]] + qi[iu[i+1,1:ny-2]] ) / 2 * ( qi[iv[i+1  ,2:ny-1  ]] + qi[iv[i,2:ny-1 ]] ) / 2 )   \
                - (1/dy) * ( - ( qi[iv[i  ,1:ny-2]] + qi[iv[i  ,2:ny-1  ]] ) / 2 * ( qi[iv[i,1:ny-2  ]] + qi[iv[i  ,2:ny-1  ]] ) / 2          \
                                + ( qi[iv[i  ,3:ny  ]] + qi[iv[i  ,2:ny-1]] ) / 2 * ( qi[iv[i,3:ny]] + qi[iv[i  ,2:ny-1]] ) / 2 )   
    
    
    # bottom inner 
    j=1
    i=np.nan
    qo[iv[1:nx-1, j]] = - (1/dx) * ( - ( qi[iu[1:nx-1,j-1]] + qi[iu[1:nx-1  ,j ]] ) / 2 * ( qi[iv[0:nx-2,j ]] + qi[iv[1:nx-1  ,j ]] ) / 2          \
                            - ( qi[iu[2:nx ,j  ]] + qi[iu[2:nx,j ]] ) / 2 * ( qi[iv[2:nx  ,j ]] + qi[iv[1:nx-1,j ]] ) / 2 )   \
               - (1/dy) * (  ( vBC_B+ qi[iv[1:nx-1  ,j ]] ) / 2 * ( vBC_B+ qi[iv[1:nx-1  ,j  ]] ) / 2          \
                            + ( qi[iv[1:nx-1  ,j+1 ]] + qi[iv[1:nx-1  ,j]] ) / 2 * ( qi[iv[1:nx-1,j+1]] + qi[iv[1:nx-1  ,j]] ) / 2 )  
            
    # right inner
    i=-1
    j=np.nan
    qo[iv[i, 2:ny-1]] = - (1/dx) * ( - ( qi[iu[i,2:ny-1  ]] + qi[iu[i  ,1:ny-2]] ) / 2 * ( qi[iv[i-1,2:ny-1  ]] + qi[iv[i  ,2:ny-1  ]] ) / 2          \
                            + ( uBC_R*vBC_R) )   \
               - (1/dy) * ( - ( qi[iv[i  ,1:ny-2]] + qi[iv[i  ,2:ny-1  ]] ) / 2 * ( qi[iv[i,1:ny-2  ]] + qi[iv[i  ,2:ny-1  ]] ) / 2          \
                            + ( qi[iv[i  ,3:ny  ]] + qi[iv[i  ,2:ny-1]] ) / 2 * ( qi[iv[i,3:ny]] + qi[iv[i  ,2:ny-1]] ) / 2 )   

    # top inner 
    j=-1
    i=np.nan
    qo[iv[1:nx-1, j]] = - (1/dx) * ( - ( qi[iu[0:nx-2,j  ]] + qi[iu[1:nx-1  ,j  ]] ) / 2 * ( qi[iv[0:nx-2,j  ]] + qi[iv[1:nx-1  ,j  ]] ) / 2          \
                                + ( qi[iu[2:nx ,j-1  ]] + qi[iu[2:nx,j  ]] ) / 2 * ( qi[iv[2:nx  ,j  ]] + qi[iv[1:nx-1,j ]] ) / 2 )   \
                - (1/dy) * ( - ( qi[iv[1:nx-1  ,j-1]] + qi[iv[1:nx-1  ,j  ]] ) / 2 * ( qi[iv[1:nx-1,j-1  ]] + qi[iv[1:nx-1  ,j  ]] ) / 2          \
                                + ( vBC_T + qi[iv[1:nx-1  ,j]] ) / 2 * ( vBC_T + qi[iv[1:nx-1  ,j]] ) / 2 )   

    ## Corners
    # bottom left   ##Needs Coding
    i = 0
    j = 1
    qo[iv[i, j]] = - (1/dx) * ( - ( uBC_L*vBC_L)         \
                                + ( qi[iu[i+1 ,j  ]] + qi[iu[i+1,j -1]] ) / 2 * ( qi[iv[i+1  ,j  ]] + qi[iv[i,j ]] ) / 2 )   \
               - (1/dy) * (  ( vBC_B+ qi[iv[i  ,j  ]] ) / 2 * ( vBC_B+ qi[iv[i  ,j  ]] ) / 2          \
                            + ( qi[iv[i  ,j+1  ]] + qi[iv[i  ,j]] ) / 2 * ( qi[iv[i,j+1]] + qi[iv[i  ,j]] ) / 2 )   

    # bottom right   ##Needs Coding
    i=-1
    j=1
    qo[iv[i, j]] = - (1/dx) * ( - ( qi[iu[i,j  ]] + qi[iu[i  ,j -1]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
                            + ( uBC_R*vBC_R) )   \
               - (1/dy) * (  ( vBC_B+ qi[iv[i  ,j  ]] ) / 2 * ( vBC_B+ qi[iv[i  ,j  ]] ) / 2          \
                            + ( qi[iv[i  ,j+1  ]] + qi[iv[i  ,j]] ) / 2 * ( qi[iv[i,j+1]] + qi[iv[i  ,j]] ) / 2 )   
    # top left   ##Needs Coding
    i=0
    j=-1
    qo[iv[i, j]] = - (1/dx) * ( - ( uBC_L*vBC_L)         \
                                + ( qi[iu[i+1 ,j  ]] + qi[iu[i+1,j -1]] ) / 2 * ( qi[iv[i+1  ,j  ]] + qi[iv[i,j ]] ) / 2 )   \
               - (1/dy) * ( - ( qi[iv[i  ,j-1]] + qi[iv[i  ,j  ]] ) / 2 * ( qi[iv[i,j-1  ]] + qi[iv[i  ,j  ]] ) / 2          \
                            + ( qi[iv[i  ,j+1  ]] + qi[iv[i  ,j]] ) / 2 * ( qi[iv[i,j+1]] + qi[iv[i  ,j]] ) / 2 )   

    ### top right   ##Needs Coding
    i=-1
    j=-1
    qo[iv[i, j]] = - (1/dx) * ( - ( qi[iu[i,j  ]] + qi[iu[i  ,j -1]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
                            + ( uBC_R*vBC_R) )   \
               - (1/dy) * ( - ( qi[iv[i  ,j-1]] + qi[iv[i  ,j  ]] ) / 2 * ( qi[iv[i,j-1  ]] + qi[iv[i  ,j  ]] ) / 2          \
                            + ( qi[iv[i  ,j+1  ]] + qi[iv[i  ,j]] ) / 2 * ( qi[iv[i,j+1]] + qi[iv[i  ,j]] ) / 2 )   
    return qo

def S_operator(qi,nu,iu,iv,nx,ny,dx,dy,dt,v):
    return qi+(dt/2)*Laplace(qi,nu,iu,iv,nx,ny,dx,dy,dt,v)


def R_operator(qi,nu,iu,iv,nx,ny,dx,dy,dt,v):
    return qi-(dt/2)*Laplace(qi,nu,iu,iv,nx,ny,dx,dy,dt,v)
    
def R_inv_operator(qi,nu,iu,iv,nx,ny,dx,dy,dt,v):
    return qi+(dt/2)*Laplace(qi,nu,iu,iv,nx,ny,dx,dy,dt,v)    

def CG_solver(Opt,b,qi,args,cg_iter):
    rhs=b
    lhs=Opt(qi,*args)
    d_old=rhs-lhs
    r_old=d_old

    for i in range(cg_iter):
        intermediate_vec=Opt(d_old,*args)
        alpha_factor=((r_old.T)@r_old)/((d_old.T)@(intermediate_vec))

        qi=qi+(alpha_factor[0,0])*d_old
        r_new=r_old-(alpha_factor[0,0])*(intermediate_vec)

        beta=((r_new.T)@(r_new))/((r_old.T)@(r_old))
        d_new=r_new+(beta[0,0])*d_old
        d_old=d_new
        r_old=r_new
    return qi
# 
def CG_solver_all(Opt,b,qi,args1,args2,args3,cg_iter):
    rhs=b
    res_list=[]
    if len(Opt)==3:
        #lhs=Opt[0](qi,*args1)
        #lhs=Opt[1](lhs,*args2)
        lhs=Opt[2](Opt[1](Opt[0](qi,*args1),*args2),*args3)
    elif len(Opt)==2:
        lhs=Opt[1](Opt[0](qi,*args1),*args2)
    elif len(Opt)==1:
        lhs=Opt[0](qi,*args1)
    
    d_old=rhs-lhs
    r_old=d_old

    for i in range(cg_iter):
        if len(Opt)==3:
            intermediate_vec=Opt[2](Opt[1](Opt[0](d_old,*args1),*args2),*args3)
        elif len(Opt)==2:
            intermediate_vec=Opt[1](Opt[0](d_old,*args1),*args2)
        elif len(Opt)==1:
            intermediate_vec=Opt[0](d_old,*args1)

        alpha_factor=((r_old.T)@r_old)/((d_old.T)@(intermediate_vec))
        
        tr=qi
        qi=qi+(alpha_factor[0,0])*d_old
        if i!=0:
            res=norm(tr-qi)/norm(tr)
            res_list.append(res)
            if res<1e-3:
                break
        r_new=r_old-(alpha_factor[0,0])*(intermediate_vec)

        beta=((r_new.T)@(r_new))/((r_old.T)@(r_old))
        d_new=r_new+(beta[0,0])*d_old
        d_old=d_new
        r_old=r_new

    return qi

def pointer_mapping(mat,flags):
    vis_mat=pd.DataFrame(mat.T)
    if flags==1:
        vis_mat=vis_mat[::-1]
    vis_mat.columns=[f"x={i}" for i in vis_mat.columns]
    vis_mat.index=[f"y={i}" for i in vis_mat.index]
    return vis_mat

def curl_operator(qi,np_,nu,ip,iu,iv,nx,ny,dx,dy,dt,v):
        ## Initialize output
    qo = np.nan*np.ones((np_,1))

    ## inner domain
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            qo[ip[i, j]] = ((- qi[iv[i, j]] + qi[iv[i+1, j]] ) / dx) - ((- qi[iu[i, j]] + qi[iu[i, j+1]] ) / dy)
    ## Edges
    ## bottom inner
    j=0
    for i in range(1,nx-1):
        qo[ip[i, j]] = ((- qi[iv[i, j]] + qi[iv[i+1, j]] ) / dx) - (( + qi[iu[i, j+1]] ) / dy)  ## -qi[iv[i, j]]

    ## top inner
    j = -1
    for i in range(1,nx-1):
        qo[ip[i, j]] = (( - qi[iv[i, j]] + qi[iv[i+1, j]] ) / dx) - (( - qi[iu[i, j]]) / dy)   ## + qi[iv[i, j+1]]

    ## left inner
    i = 0
    for j in range(1,ny-1):
        qo[ip[i, j]] = ((+ qi[iv[i+1, j]] ) / dx)  - (( - qi[iu[i, j]] + qi[iu[i, j+1]] ) / dy) ## - qi[iu[i, j]] 

    ## right inner
    i=-1
    for j in range(1,ny-1):
        qo[ip[i, j]] = ((- qi[iv[i, j]] ) / dx) - ((- qi[iu[i, j]] + qi[iu[i, j+1]] ) / dy) ##+ qi[iu[i+1, j]] 

    ## Corners
    ## bottom left (pinning)
    i = 0
    j = 0
    #qo[ip[i, j]] = (( + qi[iu[i+1, j]] ) / dx) + (( + qi[iv[i, j+1]] ) / dy)
    qo[ip[i, j]] = 0

    # bottom right (pinning)
    i=-1
    j=0
    #qo[ip[i, j]] =((- qi[iu[i, j]] ) / dx) + ( + qi[iv[i, j+1]] ) / dy)
    qo[ip[i, j]] =0

    ## top left
    i = 0
    j = -1
    qo[ip[i, j]] = ((+ qi[iv[i+1, j]] ) / dx) - (( - qi[iu[i, j]]) / dy) ## - qi[iu[i, j]]   ## + qi[iv[i, j+1]] 

    ## top right
    i=-1
    j=-1
    qo[ip[i, j]] =((- qi[iv[i, j]] ) / dx) - ((- qi[iu[i, j]] ) / dy) # ## + qi[iu[i+1, j]]  + qi[iv[i, j+1]] 
    
    return qo

def BC_Curl(uBC_L, uBC_R, vBC_T, vBC_B,np_,ip,nx,ny,dx,dy):

    ## BC vector for divergence operator: 
    #       input: BCs
    #       output: p-type (np elements)

    ## Initialize output
    bcC = np.zeros((np_, 1))

    ## Edges
    # bottom inner
    i=-1
    for j in range(1,ny-1):
        bcC[ip[i, j]] =   -vBC_B / dx 

    # top inner
    j = -1
    for i in range(1,nx-1):
        bcC[ip[i, j]] =   vBC_T / dx   # qi(iv(i, j+1))
    # left inner
    i = 0
    for j in range(1,ny-1):
        bcC[ip[i, j]] = - uBC_L / dy

    # right inner
    i=-1
    for j in range(1,ny-1):
        bcC[ip[i, j]] = uBC_R / dy


    ## Corners
    # bottom left (need pinning)
    i = 0
    j = 0
    bcC[ip[i, j]] = 0

    # bottom right
    i=-1
    j=0
    bcC[ip[i, j]] = 0
    

    # top left
    i = 0
    j = -1
    bcC[ip[i, j]] = (- uBC_L / dy) - (vBC_T / dx) # - qi[iu[i, j]] + qi(iv(i, j+1))

    # top right
    i=-1
    j=-1
    bcC[ip[i, j]] = (+ uBC_R / dy) - (vBC_T / dx) # ## + qi[iu[i+1, j]]  + qi[iv[i, j+1]] 

    return bcC
def inter_velocity(u_new,nx,ny,iu,iv,idu):
    u_new_int=np.zeros((nx*ny,1))
    v_new_int=np.zeros((nx*ny,1))

    for i in range(1,nx):
        for j in range(0,ny-1):
            u_new_int[idu[i-1,j]]=(u_new[iu[i,j]]+u_new[iu[i,j+1]])/2
    
    for i in range(0,nx-1):
        for j in range(1,ny):
            v_new_int[idu[i,j-1]]=(u_new[iv[i,j]]+u_new[iv[i+1,j]])/2
    return u_new_int,v_new_int
def explicit_matrix(A_operator,out,inp,args):

    matrix_A=[]
    for i in range(inp):
        Q=np.zeros((inp,1))

        Q[i,0]=1
        matrix_A.append(list(A_operator(Q,*args)[:,0]))
    return np.array(matrix_A).T
def CG_solver_trial(R_operator,b,u_new,args,cg_iter):
    rhs=b

    qi =[np.random.rand(u_new.shape[0],u_new.shape[1]) for i in range(4)]

    with Pool() as pool:
        lhs = pool.starmap(R_operator, [(qi[0], *args), (qi[1], *args), (qi[2], *args), (qi[3], *args)])

    d_old=rhs-lhs
    r_old=d_old

    for i in range(cg_iter):
        with Pool() as pool:
            intermediate_vec= pool.starmap(R_operator, [(d_old[0], *args), (d_old[1], *args), (d_old[2], *args), (d_old[3], *args)])
        alpha_factor=(((r_old[0].T@r_old).reshape(r_old.shape[0],1))/((d_old[0].T@intermediate_vec).reshape(r_old.shape[0],1))).reshape(r_old.shape[0],1,1)
        tr=qi
        r_new=r_old-(alpha_factor)*(np.array(intermediate_vec))
        qi=qi+(alpha_factor)*d_old
        if i!=0:
            res=(norm(tr-qi,axis=1)/(norm(tr,axis=1)))
            res[np.isnan(res)] = 0
            res_min=np.min(res[res>0])
            ind=np.where(res == res_min)[0][0]
            if res_min<1e-4:
                break
        beta=((((r_new[0].T)@(r_new)).reshape(r_new.shape[0],1)/((r_old[0].T@r_old).reshape(r_old.shape[0],1))).reshape(r_old.shape[0],1,1))
        d_new=r_new + (beta)*d_old
        d_old=d_new
        r_old=r_new
    return qi[ind]
