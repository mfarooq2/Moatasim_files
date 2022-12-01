def data_unloader(**kwargs):
    ip, iu, iv,idu = GenPointer(nx, ny)
    resume=False
    dx = 1 / nx
    dy = 1 / ny

    np_ = nx * ny
    nu = 2*nx*ny - nx - ny
    X,Y=np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny), indexing='ij')
    v=1/Re
    dt=0.79/((1/dx)+(1/dy))
    p  = np.zeros((np_, 1))
    u  = np.zeros((nu, 1))
    qi = np.zeros((nu,1))
    b  = np.zeros((nu,1))
    bcL=BC_Laplace(uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,nu,iu,iv,nx,ny,dx,dy)
    bcD = BC_Div(uBC_L, uBC_R, vBC_T, vBC_B,np_,ip,nx,ny,dx,dy)
