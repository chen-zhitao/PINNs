import numpy as np
import math
import torch
import matplotlib.pyplot as plt


# load theory results
u_theory=np.load('u_exact.npy')
u_0=-np.sin(np.pi* np.arange(-1,1.01,0.01)) # t=0.0 exact
u_theory=np.column_stack((u_0,u_theory)) # now u_theory has 4 columns


# load neural network models

D_NN_1 = torch.load('D_NN_5.pt')
D_NN_1.eval()
NN_IB_1 = torch.load('NN_IB_5.pt')
NN_IB_1.eval()
u_tilda_1=torch.load('u_tilda_5.pt')
u_tilda_1.eval()


D_NN_2 = torch.load('D_NN_6.pt')
D_NN_2.eval()
NN_IB_2 = torch.load('NN_IB_6.pt')
NN_IB_2.eval()
u_tilda_2=torch.load('u_tilda_6.pt')
u_tilda_2.eval()


D_NN_3 = torch.load('D_NN_7.pt')
D_NN_1.eval()
NN_IB_3 = torch.load('NN_IB_7.pt')
NN_IB_3.eval()
u_tilda_3=torch.load('u_tilda_7.pt')
u_tilda_3.eval()


D_NN_4 = torch.load('D_NN_4.pt')
D_NN_4.eval()
NN_IB_4 = torch.load('NN_IB_4.pt')
NN_IB_4.eval()
u_tilda_4=torch.load('u_tilda_4.pt')
u_tilda_4.eval()


# create torch tensors in the form of (t,x)
# first 201 rows are for t=0.0

xrange=np.arange(-1,1.01,0.01)
coords=torch.zeros((804,2))

coords[:201,0]=0.0
coords[201:402,0]=0.25
coords[402:603,0]=0.5
coords[603:,0]=0.75


coords[:201,1]=torch.from_numpy(np.arange(-1,1.01,0.01))
coords[201:402,1]=torch.from_numpy(np.arange(-1,1.01,0.01))
coords[402:603,1]=torch.from_numpy(np.arange(-1,1.01,0.01))
coords[603:,1]=torch.from_numpy(np.arange(-1,1.01,0.01))




# evaluate model
u_predict_1=(NN_IB_1(coords)+D_NN_1(coords)*u_tilda_1(coords)).detach().numpy()
u_predict_2=(NN_IB_2(coords)+D_NN_2(coords)*u_tilda_2(coords)).detach().numpy()
u_predict_3=(NN_IB_3(coords)+D_NN_3(coords)*u_tilda_3(coords)).detach().numpy()
u_predict_4=(NN_IB_4(coords)+D_NN_4(coords)*u_tilda_4(coords)).detach().numpy()


######################################################################
# calculate L^2 error for model 1
error_1=np.zeros(4)

# for t=0.
# denominator
D0=np.square(u_theory[:,0])
D0=np.sqrt(np.sum(D0))
# numerator
N0=np.square(u_predict_1[:201,0]-u_theory[:,0])
N0=np.sqrt(np.sum(N0))
error_1[0]=N0/D0

# for t=0.25
# denominator
D1=np.square(u_theory[:,1])
D1=np.sqrt(np.sum(D1))
# numerator
N1=np.square(u_predict_1[201:402,0]-u_theory[:,1])
N1=np.sqrt(np.sum(N1))
error_1[1]=N1/D1

# for t=0.5
# denominator
D2=np.square(u_theory[:,2])
D2=np.sqrt(np.sum(D2))
# numerator
N2=np.square(u_predict_1[402:603,0]-u_theory[:,2])
N2=np.sqrt(np.sum(N2))
error_1[2]=N2/D2


# for t=0.75
# denominator
D3=np.square(u_theory[:,3])
D3=np.sqrt(np.sum(D3))
# numerator
N3=np.square(u_predict_1[603:,0]-u_theory[:,3])
N3=np.sqrt(np.sum(N3))
error_1[3]=N3/D3
######################################################################



######################################################################
# calculate L^2 error for model 2
error_2=np.zeros(4)

# for t=0.
# denominator
D0=np.square(u_theory[:,0])
D0=np.sqrt(np.sum(D0))
# numerator
N0=np.square(u_predict_2[:201,0]-u_theory[:,0])
N0=np.sqrt(np.sum(N0))
error_2[0]=N0/D0

# for t=0.25
# denominator
D1=np.square(u_theory[:,1])
D1=np.sqrt(np.sum(D1))
# numerator
N1=np.square(u_predict_2[201:402,0]-u_theory[:,1])
N1=np.sqrt(np.sum(N1))
error_2[1]=N1/D1

# for t=0.5
# denominator
D2=np.square(u_theory[:,2])
D2=np.sqrt(np.sum(D2))
# numerator
N2=np.square(u_predict_2[402:603,0]-u_theory[:,2])
N2=np.sqrt(np.sum(N2))
error_2[2]=N2/D2


# for t=0.75
# denominator
D3=np.square(u_theory[:,3])
D3=np.sqrt(np.sum(D3))
# numerator
N3=np.square(u_predict_2[603:,0]-u_theory[:,3])
N3=np.sqrt(np.sum(N3))
error_2[3]=N3/D3
######################################################################



######################################################################
# calculate L^2 error for model 3
error_3=np.zeros(4)

# for t=0.
# denominator
D0=np.square(u_theory[:,0])
D0=np.sqrt(np.sum(D0))
# numerator
N0=np.square(u_predict_3[:201,0]-u_theory[:,0])
N0=np.sqrt(np.sum(N0))
error_3[0]=N0/D0

# for t=0.25
# denominator
D1=np.square(u_theory[:,1])
D1=np.sqrt(np.sum(D1))
# numerator
N1=np.square(u_predict_3[201:402,0]-u_theory[:,1])
N1=np.sqrt(np.sum(N1))
error_3[1]=N1/D1

# for t=0.5
# denominator
D2=np.square(u_theory[:,2])
D2=np.sqrt(np.sum(D2))
# numerator
N2=np.square(u_predict_3[402:603,0]-u_theory[:,2])
N2=np.sqrt(np.sum(N2))
error_3[2]=N2/D2


# for t=0.75
# denominator
D3=np.square(u_theory[:,3])
D3=np.sqrt(np.sum(D3))
# numerator
N3=np.square(u_predict_3[603:,0]-u_theory[:,3])
N3=np.sqrt(np.sum(N3))
error_3[3]=N3/D3
######################################################################




######################################################################
# calculate L^2 error for model 4
error_4=np.zeros(4)

# for t=0.
# denominator
D0=np.square(u_theory[:,0])
D0=np.sqrt(np.sum(D0))
# numerator
N0=np.square(u_predict_4[:201,0]-u_theory[:,0])
N0=np.sqrt(np.sum(N0))
error_4[0]=N0/D0

# for t=0.25
# denominator
D1=np.square(u_theory[:,1])
D1=np.sqrt(np.sum(D1))
# numerator
N1=np.square(u_predict_4[201:402,0]-u_theory[:,1])
N1=np.sqrt(np.sum(N1))
error_4[1]=N1/D1

# for t=0.5
# denominator
D2=np.square(u_theory[:,2])
D2=np.sqrt(np.sum(D2))
# numerator
N2=np.square(u_predict_4[402:603,0]-u_theory[:,2])
N2=np.sqrt(np.sum(N2))
error_4[2]=N2/D2


# for t=0.75
# denominator
D3=np.square(u_theory[:,3])
D3=np.sqrt(np.sum(D3))
# numerator
N3=np.square(u_predict_4[603:,0]-u_theory[:,3])
N3=np.sqrt(np.sum(N3))
error_4[3]=N3/D3
######################################################################



"""
# making plots
# with model1

plt.rcParams["figure.figsize"] = (15,3)
plt.rcParams['text.usetex'] = True
fig, axis=plt.subplots(1,4)


axis[0].plot(xrange,-np.sin(np.pi* xrange),linewidth='3',label='Exact')
axis[0].plot(xrange,u_predict_1[:201,0],'--',linewidth='3',label='Precition')
#axis[0].set_aspect(1)
axis[0].set_aspect('equal', 'box')
axis[0].set_xlabel(r'$x$')
axis[0].set_ylabel(r'$u(t,x)$')
axis[0].set_title(r'$t=0$')


axis[1].plot(xrange,u_theory[:,1],linewidth='3',label='Exact')
axis[1].plot(xrange,u_predict_1[201:402,0],'--',linewidth='3',label='Precition')
#axis[0].set_aspect(1)
axis[1].set_aspect('equal', 'box')
axis[1].set_xlabel(r'$x$')
axis[1].set_ylabel(r'$u(t,x)$')
axis[1].set_title(r'$t=0.25$')

axis[2].plot(xrange,u_theory[:,2],linewidth='3',label='Exact')
axis[2].plot(xrange,u_predict_1[402:603,0],'--',linewidth='3',label='Precition')
#axis[1].set_aspect(1)
axis[2].set_aspect('equal', 'box')
axis[2].set_xlabel(r'$x$')
axis[2].set_ylabel(r'$u(t,x)$')
axis[2].set_title(r'$t=0.50$')
                    
                    
axis[3].plot(xrange,u_theory[:,3],linewidth='3',label='Exact')
axis[3].plot(xrange,u_predict_1[603:,0],'--',linewidth='3',label='Precition')
#axis[2].set_aspect(1)
axis[3].set_aspect('equal', 'box')
axis[3].set_ylim([-1,1])
axis[3].set_yticks([-1.0,-0.5,0.0,0.5,1.0],['-1.0','-0.5','0.0','0.5','1.0'])
axis[3].set_xlabel(r'$x$')
axis[3].set_ylabel(r'$u(t,x)$')
axis[3].set_title(r'$t=0.75$')

fig.legend(['Exact','Prediction'],bbox_to_anchor=(0.52, -0.2),ncol=2,loc = 'lower center')

#plt.savefig('Burgers_3NN.pdf',bbox_inches='tight') 
"""


