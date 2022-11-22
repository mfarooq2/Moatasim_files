import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

def Func_GaussElimination(A,b):
    n=A.shape[0]
    G=np.concatenate((A,b),axis = 1)
    fac=np.zeros((n,1))
    for i in range(n):
        if G[i][i]==0:
            for k in range(i+1,n):
                if (G[k][i]!=0):
                    tmp=G[k,:]
                    G[k,:]=G[i,:]
                    G[i,:]=tmp
                    break
        fac[i+1:n]=(-G[i+1:n,i].reshape((len(G[i+1:n,i]),1))/G[i,i])
        G[i+1:n,:]= G[i+1:n,:]+(fac[i+1:n])@(G[i,:].reshape((len(G[i,:]),1))).T
    x=np.zeros((n,1))
    x[n-1,0]=G[n-1,n]/G[n-1,n-1]
    for num in range(n-2, -1, -1):
        x[num][0] = (G[num,n]-(G[num,num+1:n].reshape((len(G[num,num+1:n]),1))).T@x[num+1:n])/G[num][num]
    return x

def plotter(xlist,f_new_list,interv,dat_acq,p,q):
    # create a figure and axes
    fig = plt.figure(figsize=(12,5))
    ax2 = plt.subplot(1,2,1)   
    ax2.set_xlim((0,1))
    ax2.set_ylim((p,q))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Phase Plane')
    pt1, = ax2.plot([], [], 'g', ms=8)

    txt_title = ax2.set_title('')
    line4, = ax2.plot([], [], 'r', lw=0.5)
    # animation function. This is called sequentially
    def drawframe(n):
        line4.set_data(xlist[n],f_new_list[n])
        line4.set_label(f"time={n*dat_acq}")
        legend = plt.legend()
        return (line4,legend)
    # blit=True re-draws only the parts that have changed.
    anim = animation.FuncAnimation(fig, drawframe, frames=len(f_new_list), interval=interv, blit=True)
    return anim

def plotter_animation_contour(xv, yv, Solution_Matrix,interv,dat_acq,p,q):
    # create a figure and axes
    fig = plt.figure(figsize=(12,5))
    ax2 = plt.subplot(1,2,1)   
    ax2.set_xlim((0,1))
    ax2.set_ylim((p,q))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Phase Plane')
    pt1, = ax2.plot([], [], 'g', ms=8)

    txt_title = ax2.set_title('')
    line4, = ax2.plot([], [], 'r', lw=0.5)
    # animation function. This is called sequentially
    def drawframe(n):
        cp = ax2.contourf(xv, yv, Solution_Matrix[n])
        if n==99:
            fig.colorbar(cp)
        line4.set_label(f"time={n*dat_acq}")
        return (line4,fig)
    # blit=True re-draws only the parts that have changed.
    anim = animation.FuncAnimation(fig, drawframe, frames=len(Solution_Matrix), interval=interv, blit=True)
    return anim