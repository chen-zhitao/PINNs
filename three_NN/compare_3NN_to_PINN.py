import numpy as np
import math
import torch
import matplotlib.pyplot as plt


# load theory results
u_theory=np.load('u_exact.npy')
u_0=-np.sin(np.pi* np.arange(-1,1.01,0.01)) # t=0.0 exact
u_theory=np.column_stack((u_0,u_theory)) # now u_theory has 4 columns

# load best PINN result
u_PINN=np.load('u_pinn_7.npy')

# load worst 3NN result
u_3NN=np.load('u_3NN_5.npy')

xrange=np.arange(-1,1.01,0.01)


# making plots

plt.rcParams["figure.figsize"] = (15,3)
plt.rcParams['text.usetex'] = True
fig, axis=plt.subplots(1,4)


axis[0].plot(xrange,-np.sin(np.pi* xrange),linewidth='3',label='Exact')
axis[0].plot(xrange,u_PINN[:201,0],':',linewidth='3',label='PINN prediction')
axis[0].plot(xrange,u_3NN[:201,0],'--',linewidth='3',label='3NN prediction')
#axis[0].set_aspect(1)
axis[0].set_aspect('equal', 'box')
axis[0].set_xlabel(r'$x$')
axis[0].set_ylabel(r'$u(t,x)$')
axis[0].set_title(r'$t=0$')


axis[1].plot(xrange,u_theory[:,1],linewidth='3',label='Exact')
axis[1].plot(xrange,u_PINN[201:402,0],':',linewidth='3',label='PINN prediction')
axis[1].plot(xrange,u_3NN[201:402,0],'--',linewidth='3',label='3NN prediction')
#axis[0].set_aspect(1)
axis[1].set_aspect('equal', 'box')
axis[1].set_xlabel(r'$x$')
axis[1].set_ylabel(r'$u(t,x)$')
axis[1].set_title(r'$t=0.25$')

axis[2].plot(xrange,u_theory[:,2],linewidth='3',label='Exact')
axis[2].plot(xrange,u_PINN[402:603,0],':',linewidth='3',label='PINN prediction')
axis[2].plot(xrange,u_3NN[402:603,0],'--',linewidth='3',label='3NN prediction')
#axis[1].set_aspect(1)
axis[2].set_aspect('equal', 'box')
axis[2].set_xlabel(r'$x$')
axis[2].set_ylabel(r'$u(t,x)$')
axis[2].set_title(r'$t=0.50$')
                    
                    
axis[3].plot(xrange,u_theory[:,3],linewidth='3',label='Exact')
axis[3].plot(xrange,u_PINN[603:,0],':',linewidth='3',label='PINN prediction')
axis[3].plot(xrange,u_3NN[603:,0],'--',linewidth='3',label='3NN prediction')
#axis[2].set_aspect(1)
axis[3].set_aspect('equal', 'box')
axis[3].set_ylim([-1,1])
axis[3].set_yticks([-1.0,-0.5,0.0,0.5,1.0],['-1.0','-0.5','0.0','0.5','1.0'])
axis[3].set_xlabel(r'$x$')
axis[3].set_ylabel(r'$u(t,x)$')
axis[3].set_title(r'$t=0.75$')

fig.legend(['Exact','PINN Prediction','3NN Prediction'],bbox_to_anchor=(0.52, -0.2),ncol=3,loc = 'lower center')

plt.savefig('Burgers_3NN.pdf',bbox_inches='tight') 


