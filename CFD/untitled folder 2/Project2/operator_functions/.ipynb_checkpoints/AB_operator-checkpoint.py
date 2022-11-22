import numpy as np
from operator_functions.pointer_func import pointer_vector
def A_operator(Q,Nx,Ny,dx,dy,dt,alpha):
    iQ=pointer_vector(Nx, Ny)

    ## Initialize output 
    Y=np.nan*np.ones((len(Q),1))

    ## inner domain
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            Y[iQ[i, j]] = Q[iQ[i, j]]+(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) 
            #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]+(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) )
    ## Edges
    # left (Dirichlet)
    i = 0 ;
    for j in range(1,Ny-1):
            Y[iQ[i, j]] = Q[iQ[i, j]]+(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2)))
            #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]+(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))))
    # top (Dirichlet)
    j = -1
    for i in range(1,Nx-1):
            Y[iQ[i, j]] = Q[iQ[i, j]]+(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] ) /(dy**2)))
            #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]+(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] ) /(dy**2))))

    # Bottom (Neuman)
    j = 0
    for i in range(1,Nx-1):
            Y[iQ[i, j]] = Q[iQ[i, j]]+(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2)))
            #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]+(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))))
    # Right (Neuman)
    i=-1
    for j in range(1,Ny-1):
        Y[iQ[i, j]] = Q[iQ[i, j]]+(alpha*dt/2)*(( (Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2)))
        #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]+(alpha*dt/2)*(( (Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))))
        
    ## Corners
    # bottom left
    i = 0
    j = 0
    Y[iQ[i, j]] = Q[iQ[i, j]]+(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) 
    #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]+(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) )
    ## bottom right
    i=-1
    j=0

    Y[iQ[i, j]] = Q[iQ[i, j]]+(alpha*dt/2)*(((Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) 
    #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]+(alpha*dt/2)*(((Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) )
    ## top left
    j=-1
    i=0
    Y[iQ[i, j]] = Q[iQ[i, j]]+(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] ) /(dy**2))) 

    ## top right
    i=-1
    j=-1
    Y[iQ[i, j]] = Q[iQ[i, j]]+(alpha*dt/2)*(((Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ (( Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] )/(dy**2)))
    #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]+(alpha*dt/2)*(((Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ (( Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] )/(dy**2))))
    return Y
def B_operator(Q,Nx,Ny,dx,dy,dt,alpha):
    iQ=pointer_vector(Nx, Ny)

    ## Initialize output 
    Y=np.nan*np.ones((len(Q),1))

    ## inner domain
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            Y[iQ[i, j]] = Q[iQ[i, j]]-(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) 
            #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]-(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) )
    ## Edges
    # left (Dirichlet)
    i = 0 ;
    for j in range(1,Ny-1):
            Y[iQ[i, j]] = Q[iQ[i, j]]-(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2)))
            #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]-(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))))
    # top (Dirichlet)
    j = -1
    for i in range(1,Nx-1):
            Y[iQ[i, j]] = Q[iQ[i, j]]-(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] ) /(dy**2)))
            #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]-(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] ) /(dy**2))))

    # Bottom (Neuman)
    j = 0
    for i in range(1,Nx-1):
            Y[iQ[i, j]] = Q[iQ[i, j]]-(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2)))
            #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]-(alpha*dt/2)*(((Q[iQ[i-1, j]] - 2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))))

    # Right (Neuman)
    i=-1
    for j in range(1,Ny-1):
        Y[iQ[i, j]] = Q[iQ[i, j]]-(alpha*dt/2)*(( (Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2)))
        #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]-(alpha*dt/2)*(( (Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))))
        
    ## Corners
    # bottom left
    i = 0
    j = 0
    Y[iQ[i, j]] = Q[iQ[i, j]]-(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) 

    #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]-(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) )
    ## bottom right
    i=-1
    j=0

    Y[iQ[i, j]] = Q[iQ[i, j]]-(alpha*dt/2)*(((Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) 
    #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]-(alpha*dt/2)*(((Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ ((-1*Q[iQ[i, j]] + Q[iQ[i, j+1]] ) /(dy**2))) )
    ## top left
    j=-1
    i=0

    Y[iQ[i, j]] = Q[iQ[i, j]]-(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] ) /(dy**2))) 
    #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]-(alpha*dt/2)*(( (-2*Q[iQ[i, j]] + Q[iQ[i+1, j]] ) /(dx**2))+ ((Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] ) /(dy**2))) )
    ## top right
    i=-1
    j=-1
    Y[iQ[i, j]] = Q[iQ[i, j]]-(alpha*dt/2)*(((Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ (( Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] )/(dy**2)))
    #Y = Y.at[[iQ[i, j]]].set(Q[iQ[i, j]]-(alpha*dt/2)*(((Q[iQ[i-1, j]] - Q[iQ[i, j]] ) /(dx**2))+ (( Q[iQ[i, j-1]] - 2*Q[iQ[i, j]] )/(dy**2))))
    return Y



def uBC_Laplace(Nx,Ny,dx,dy,BCB,BCL,BCT,BCR,dt,alpha):
    iQ=pointer_vector(Nx, Ny)

    uBC  = np.zeros((Nx*Ny, 1)) ;

    ## Edges
    #left (Dirichlet)
    i = 0
    for j in range(1,Ny-1):
        uBC[iQ[i, j]] = alpha*dt*BCL[j] / (dx**2)
        #uBC = uBC.at[iQ[i, j]].set(alpha*dt*BCL[j] / (dx**2))

    #bottom (Neuman)
    j = 0
    for i in range(1,Nx-1):
        uBC[iQ[i, j]] = alpha*dt*-1*BCB[i] / (dy)
        #uBC = uBC.at[iQ[i, j]].set(alpha*dt*-1*BCB[i] / (dy))
    # right(Neuman)
    i = -1
    for j in range(1,Ny-1):
        uBC[iQ[i, j]] = alpha*dt*BCR[j] / (dx)
        #uBC = uBC.at[iQ[i, j]].set(alpha*dt*BCR[j] / (dx))
    # top(Dirichlet)
    j = -1
    for i in range(1,Nx-1):
        uBC[iQ[i, j]] = alpha*dt*BCT[i] / (dy**2)
        #uBC = uBC.at[iQ[i, j]].set(alpha*dt*BCT[i] / (dy**2))

    ## Corners
    # bottom left (D-N)
    i = 0 ;
    j = 0 ;
    uBC[iQ[i, j]] = alpha*dt*((BCL[i] /(dx**2)) - (BCB[j] /(dy)))
    #uBC = uBC.at[iQ[i, j]].set(alpha*dt*((BCL[i] /(dx**2)) - (BCB[j] /(dy))))
    # bottom right (N-N)
    i = -1
    j = 0 
    uBC[iQ[i, j]] = alpha*dt*((BCR[i] /(dx)) - (BCB[j] /(dy)))
    #uBC = uBC.at[iQ[i, j]].set(alpha*dt*((BCR[i] /(dx)) - (BCB[j] /(dy))))
    # top left (D-D)
    i = 0
    j = -1 
    uBC[iQ[i, j]] = alpha*dt*((BCL[i] /(dx**2)) + (BCT[j] /(dy**2)))
    #uBC = uBC.at[iQ[i, j]].set(alpha*dt*((BCL[i] /(dx**2)) + (BCT[j] /(dy**2))))
    # top right (N-D)
    i=-1
    j=-1
    uBC[iQ[i, j]] = alpha*dt*((BCR[i] /(dx)) + (BCT[j] /(dy**2)))
    #uBC = uBC.at[iQ[i, j]].set((BCR[i] /(dx)) + (BCT[j] /(dy**2)))
    return uBC
def b_vector_creator(b,Nx,Ny,dx,dy,BCB,BCL,BCT,BCR,dt,alpha):
    b_vector=np.zeros((Nx*Ny,1))
    iQ=pointer_vector(Nx, Ny)
    for i in range(0,Nx):
        for j in range(0,Ny):
            b_vector[iQ[i, j]]  = b[i,j]
            #x = b_vector.at[iQ[i, j]].set(b[i,j])
    return b_vector