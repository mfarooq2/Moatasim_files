import numpy as np
def GenPointer(nx, ny):
    ## Memory allocation
    ip = np.nan*np.ones((nx,ny))
    iu = np.nan*np.ones((nx,ny))
    iv = np.nan*np.ones((nx,ny))

    ## Pointer matrix for P
    id_p = 0 # index to be used in vector variable P
    for i in range(0,nx):
        for j in range(0,ny):
            ip[i, j] = id_p
            id_p = id_p + 1 

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
    return ip.astype(int),iu.astype(int),iv.astype(int)

def Grad(qi,np_,nu,nx,ny,dx,dy,iu,iv,ip):
    ## Gradient operator: 
    #       input: p-type (np elements)
    #       output: u-type (nu elements)
    ## Input size check:
    qo = np.nan*np.ones((nu,1))

    ## inner domain
    ## x-direction gradient
    for i in range(1,nx):
        for j in range(0,ny):
            qo[iu[i, j]] = ( -qi[ip[i-1, j]] + qi[ip[i, j]] ) / dx

    ## y-direction gradient
    for i in range(0,nx):
        for j in range(1,ny):
            qo[iv[i, j]] = ( -qi[ip[i, j-1]] + qi[ip[i, j]] ) / dy
    return qo

def Div(qi,np_,nu,nx,ny,dx,dy,iu,iv,ip):

    ## Initialize output
    qo = np.nan*np.ones((np_,1))

    ## inner domain
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            qo[ip[i, j]] = ((- qi[iu[i, j]] + qi[iu[i+1, j]] ) / dx) + ((- qi[iv[i, j]] + qi[iv[i, j+1]] ) / dy)
    ## Edges
    ## bottom inner
    j=0
    for i in range(1,nx-1):
        qo[ip[i, j]] = ((- qi[iu[i, j]] + qi[iu[i+1, j]] ) / dx) + (( + qi[iv[i, j+1]] ) / dy)  ## -qi[iv[i, j]]

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

def Laplace(qi,nu,iu,iv,nx,ny,dx,dy ):
    
    ## Laplace operator: 
    #       input: u-type (nu elements)
    #       output: u-type (nu elements)

    ## Initialize output
    qo=np.nan*np.ones((nu,1))

    ## 1. ex-Component
    ## inner domain
    for i in range(2,nx-1):
        for j in range(1,ny-1):
            qo[iu[i, j]] = (+qi[iu[i-1, j]] -2*qi[iu[i, j]] + qi[iu[i+1, j]] ) / (dx**2) + ( +qi[iu[i, j-1]] -2*qi[iu[i, j]] + qi[iu[i, j+1]] ) / (dy**2)

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
    for i in range(1,nx-1):
        for j in range(2,ny-1):
            qo[iv[i, j]] = (+qi[iv[i-1, j]] -2*qi[iv[i, j]] + qi[iv[i+1, j]] ) / (dx**2) + ( +qi[iv[i, j-1]] -2*qi[iv[i, j]] + qi[iv[i, j+1]] ) / (dy**2)


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

def Adv(qi, uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,nu,iu,iv,nx,ny,dx,dy):

    
    
    
    ## advection operator (BC embedded): -\nabla \cdot (uu) 
    ##      input: u-type (nu elements)
    ##       output: u-type (nu elements)
    #


    ## Initialize output
    qo = np.nan*np.ones((nu,1))

    ## 1. U-Component
    ## inner domain
    for i in range(2,nx-2): 
        for j in range(1,ny-1):
            qo[iu[i, j]] = - (1/dx) * ( - ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2          \
                                            + ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 )   \
                               - (1/dy) * ( - ( qi[iu[i  ,j-1]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
                                            + ( qi[iu[i  ,j  ]] + qi[iu[i  ,j+1]] ) / 2 * ( qi[iv[i-1,j+1]] + qi[iv[i  ,j+1]] ) / 2 )                 

    ## Edges
    ## left inner 
    i = 1
    for j in range(2,ny-1):
        qo[iu[i, j]] = - ( - ( uBC_L           + qi[iu[i  ,j  ]] ) / 2 * ( uBC_L           + qi[iu[i  ,j  ]] ) / 2          \
                               + ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 ) / dx  \
                           - ( - ( qi[iu[i  ,j-1]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
                               + ( qi[iu[i  ,j  ]] + qi[iu[i  ,j+1]] ) / 2 * ( qi[iv[i-1,j+1]] + qi[iv[i  ,j+1]] ) / 2 ) / dy                

    # bottom inner
    j = 0
    for i in range(2,nx-1):
        qo[iu[i, j]] = - ( - ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2          \
                               + ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 ) / dx   \
                           - ( -   uBC_B                                   *   vBC_B                                            \
                               + ( qi[iu[i  ,j  ]] + qi[iu[i  ,j+1]] ) / 2 * ( qi[iv[i-1,j+1]] + qi[iv[i  ,j+1]] ) / 2 ) / dy      

    # right inner
    i=-1
    for j in range(1,ny-1):
        qo[iu[i, j]] = - (1/dx) * ( - ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2         \
                                            + ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 )   \
                               - (1/dy) * ( - ( qi[iu[i  ,j-1]] +uBC_R ) / 2 * ( qi[iv[i-1,j  ]] + uBC_R ) / 2          \
                                            + ( qi[iu[i  ,j  ]] + qi[iu[i  ,j+1]] ) / 2 * ( qi[iv[i-1,j+1]] + qi[iv[i  ,j+1]] ) / 2 )      

    # top inner
    j=-1
    for i in range(2,nx-1):
        qo[iu[i, j]] = - (1/dx) * ( - ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2         \
                                            + ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iu[i  ,j  ]] + qi[iu[i+1,j  ]] ) / 2 )   \
                               - (1/dy) * ( - ( qi[iu[i  ,j-1]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
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
    for i in range(1,nx-1): 
        for j in range(2,ny-1):
            qo[iv[i, j]] = - (1/dx) * ( - ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
                                        + ( qi[iu[i+1 ,j-1  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iv[i+1  ,j  ]] + qi[iv[i,j ]] ) / 2 )   \
                           - (1/dy) * ( - ( qi[iv[i  ,j-1]] + qi[iv[i  ,j  ]] ) / 2 * ( qi[iv[i,j-1  ]] + qi[iv[i  ,j  ]] ) / 2          \
                                        + ( qi[iv[i  ,j+1  ]] + qi[iv[i  ,j]] ) / 2 * ( qi[iv[i,j+1]] + qi[iv[i  ,j]] ) / 2 )   

    ## Edges
    # left inner
    i=0
    for j in range(2,ny-1):
        qo[iv[i, j]] = - (1/dx) * ( - ( uBC_L*vBC_L)         \
                                + ( qi[iu[i+1 ,j  ]] + qi[iu[i+1,j -1]] ) / 2 * ( qi[iv[i+1  ,j  ]] + qi[iv[i,j ]] ) / 2 )   \
                - (1/dy) * ( - ( qi[iv[i  ,j-1]] + qi[iv[i  ,j  ]] ) / 2 * ( qi[iv[i,j-1  ]] + qi[iv[i  ,j  ]] ) / 2          \
                                + ( qi[iv[i  ,j+1  ]] + qi[iv[i  ,j]] ) / 2 * ( qi[iv[i,j+1]] + qi[iv[i  ,j]] ) / 2 )   
    
    
    # bottom inner 
    j=1
    for i in range(1,nx-1):
            qo[iv[i, j]] = - (1/dx) * ( - ( qi[iu[i,j-1]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
                            - ( qi[iu[i+1 ,j-1  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iv[i+1  ,j  ]] + qi[iv[i,j ]] ) / 2 )   \
               - (1/dy) * (  ( vBC_B+ qi[iv[i  ,j  ]] ) / 2 * ( vBC_B+ qi[iv[i  ,j  ]] ) / 2          \
                            + ( qi[iv[i  ,j+1  ]] + qi[iv[i  ,j]] ) / 2 * ( qi[iv[i,j+1]] + qi[iv[i  ,j]] ) / 2 )   

    # right inner
    i=-1
    for j in range(2,ny-1):
        qo[iv[i, j]] = - (1/dx) * ( - ( qi[iu[i,j  ]] + qi[iu[i  ,j -1]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
                            + ( uBC_R*vBC_R) )   \
               - (1/dy) * ( - ( qi[iv[i  ,j-1]] + qi[iv[i  ,j  ]] ) / 2 * ( qi[iv[i,j-1  ]] + qi[iv[i  ,j  ]] ) / 2          \
                            + ( qi[iv[i  ,j+1  ]] + qi[iv[i  ,j]] ) / 2 * ( qi[iv[i,j+1]] + qi[iv[i  ,j]] ) / 2 )   

    # top inner 
    j=-1
    for i in range(1,nx-1):
        qo[iv[i, j]] = - (1/dx) * ( - ( qi[iu[i-1,j  ]] + qi[iu[i  ,j  ]] ) / 2 * ( qi[iv[i-1,j  ]] + qi[iv[i  ,j  ]] ) / 2          \
                                + ( qi[iu[i+1 ,j-1  ]] + qi[iu[i+1,j  ]] ) / 2 * ( qi[iv[i+1  ,j  ]] + qi[iv[i,j ]] ) / 2 )   \
                - (1/dy) * ( - ( qi[iv[i  ,j-1]] + qi[iv[i  ,j  ]] ) / 2 * ( qi[iv[i,j-1  ]] + qi[iv[i  ,j  ]] ) / 2          \
                                + ( vBC_T + qi[iv[i  ,j]] ) / 2 * ( vBC_T + qi[iv[i  ,j]] ) / 2 )   

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
    return qi+(dt/2)*Laplace(qi,nu,iu,iv,nx,ny,dx,dy)


def R_operator(qi,nu,iu,iv,nx,ny,dx,dy,dt,v):
    return qi-(dt/2)*Laplace(qi,nu,iu,iv,nx,ny,dx,dy)
    
def R_inv_operator(qi,nu,iu,iv,nx,ny,dx,dy,dt,v):
    return qi+(dt/2)*Laplace(qi,nu,iu,iv,nx,ny,dx,dy)    


def CG_solver(Opt,b,qi,args,dt,v,cg_iter):
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

def pointer_mapping(mat):
    vis_mat=pd.DataFrame(mat)
    vis_mat.columns=[f"x={i}" for i in vis_mat.columns]
    vis_mat.index=[f"y={len(vis_mat.index)-(i+1)}" for i in vis_mat.index]
    return vis_mat
    