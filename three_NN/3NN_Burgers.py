#!/usr/bin/env python
# coding: utf-8

# In[153]:


import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set data type
DTYPE=torch.float32

# Set constants
pi=torch.pi
#viscosity = .01/pi
viscosity=0.001

# Define initial condition
def fun_u_0(x):
    return -torch.sin(pi * x)

# Define boundary condition
def fun_u_b(t, x):
    n = x.shape[0]
    return torch.zeros((n,1), dtype=DTYPE)

# Define residual of the PDE
def fun_r(t, x, u, u_t, u_x, u_xx):
    return u_t + u * u_x - viscosity * u_xx


# In[154]:


# Set number of data points
N_0 = 500
N_b = 200
N_r = 10000

# Set boundary
tmin = 0.
tmax = 1.
xmin = -1.
xmax = 1.



# Lower bounds
lb = torch.tensor([tmin, xmin], dtype=DTYPE)
# Upper bounds
ub = torch.tensor([tmax, xmax], dtype=DTYPE)

# Set random seed for reproducible results
torch.manual_seed(0)


# In[155]:


# sample initial value
t_0=torch.ones((N_0,1),dtype=DTYPE)*lb[0]
x_0=torch.rand((N_0,1),dtype=DTYPE)

# scale x_0 to between x_min and x_max
x_0=x_0*(ub[1]-lb[1]) + lb[1]

# concatenate t_0 and x_0 into an N_0 x 2 matrix
X_0 = torch.cat([t_0, x_0], axis=1)


# Evaluate intitial condition at x_0
u_0 = fun_u_0(x_0)

#u_0.shape is torch.Size([50, 1])


# In[156]:


# Boundary data
t_b = torch.rand((N_b,1), dtype=DTYPE)
t_b = t_b*(ub[0] - lb[0])+lb[0]
x_b = lb[1] + (ub[1] - lb[1]) * torch.bernoulli(torch.ones((N_b,1))-1/2)
# or we can just create two N_b/2 by 1 tensors, one for each boundary


# concatenate to an N_b x 2 matrix
X_b = torch.cat([t_b, x_b], axis=1)


# Evaluate boundary condition at (t_b,x_b)
u_b = fun_u_b(t_b, x_b)


# In[157]:


X_b.shape


# In[158]:


# Initial and boundary data are only used to train
# the NN for initial and boundary condition


# In[159]:


# Combining initial and boundary conditions
# X_ib is (N_0 + N_b) x 2
# u_ib is (N_0 + N_b) x 1

X_ib=torch.cat([X_0, X_b],axis=0)
u_ib=torch.cat([u_0, u_b],axis=0)


# In[160]:


# Draw collocation points
t_r = torch.rand((N_r,1),dtype=DTYPE)
t_r = t_r*(ub[0]-lb[0])+lb[0]
x_r = torch.rand((N_r,1),dtype=DTYPE)
x_r = x_r*(ub[1]-lb[1])+lb[1]

X_r = torch.cat([t_r, x_r], axis=1)

# a small fraction of the collocation points are used to train the distance function
# the rest are for the PINN loss function


# In[161]:


X_r.shape


# In[162]:


# Define the three NN

# D(x,t) distance function


def D_NN_constr(num_hidden_layers=2, num_neurons=5):
    layers = []
    layers.append(nn.Linear(2, num_neurons))
    layers.append(nn.Tanh())
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(num_neurons,num_neurons))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(num_neurons, 1))

    model = nn.Sequential(*layers)
    return model


# In[163]:


# a NN to learn the initial and boundary conditions

def IB_constr(num_hidden_layers=2, num_neurons=5):
    layers = []
    layers.append(nn.Linear(2, num_neurons))
    layers.append(nn.Tanh())
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(num_neurons,num_neurons))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(num_neurons, 1))

    model = nn.Sequential(*layers)
    return model


# In[164]:


# u_tilda(x,t) is the main NN

def u_tilda_constr(num_hidden_layers=5, num_neurons=12):
    layers = []
    layers.append(nn.Linear(2, num_neurons))
    layers.append(nn.Tanh())
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(num_neurons,num_neurons))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(num_neurons, 1))

    model = nn.Sequential(*layers)
    return model


# In[182]:


# compute residual in the bulk to train u_tilda
# X_r is a (N_r x 2) torch tensor
# we need to input the other two networks


def get_r(u_net,D_net,IB_net, X_r):
    
    t, x = X_r[:, 0:1], X_r[:,1:2]
    # this way of slicing makes t and x both (N_r x 1) tensor
    # and NOT a [N_r] vector
    
    t.requires_grad_(requires_grad=True)
    x.requires_grad_(requires_grad=True)
    X = torch.stack([t[:,0], x[:,0]], axis=1)
    
    u = IB_net(X) + D_net(X) * u_net(X)
    #u = IB_net(X)*torch.exp(-15.*D_net(X)) + D_net(X) * u_net(X)
    
    # torch.stack([t[:,0], x[:,0]], axis=1) has shape (N_r x 2)
    # u has shape (N_r x 1)
    
    u_t=torch.autograd.grad(u,t,torch.ones_like(u),create_graph=True)[0]
    u_x=torch.autograd.grad(u,x,torch.ones_like(u),create_graph=True)[0]
    u_xx=torch.autograd.grad(u_x,x,torch.ones_like(u_x),create_graph=True)[0]
    # all derivatives have shape (N_r x 1) because of torch.ones_like(u)
    
    return fun_r(t, x, u, u_t, u_x, u_xx)


# In[166]:


# Define a dataset with label
# used to train distance AND initial and boundary condition

class Dataset_label(Dataset):
    
    def __init__(self,data,label):
        self.data = data  
        self.label = label 
        self.N = len(data)  
        
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    
    def __len__(self):
        return self.N

    
# Define a dataset WITHOUT label
# Used to train u_tilta

class Dataset_nolabel(Dataset):
    
    def __init__(self,data):
        self.data = data 
        self.N = len(data)  
        
    def __getitem__(self,index):
        return self.data[index]
    
    def __len__(self):
        return self.N


# In[167]:


# Training distance D(x,t)

# preparing data
# use 50 points from initial, 50 from boundary, and 50 from collocation as training data

X_D_tr=torch.cat([X_0[0:50, :], X_b[0:50, :], X_r[0:50,:]],axis=0)

# creating label
# for each (t,x), D(x,t)=min(t-tmin, x-xmin, xmax-x)

d_tr=torch.zeros((150,1),dtype=DTYPE)

for i in range(150):
    t=X_D_tr[i,0]
    x=X_D_tr[i,1]
    d_tr[i,:]=min([t-tmin, x-xmin, xmax-x])
    
    
# Create validation data
# 10 points from initial, 10 points from boundary, 10 from collocation

X_D_val=torch.cat([X_0[50:60, :], X_b[50:60, :], X_r[50:60,:]],axis=0)

d_val=torch.zeros((30,1),dtype=DTYPE)

for i in range(30):
    t=X_D_val[i,0]
    x=X_D_val[i,1]
    d_val[i,:]=min([t-tmin, x-xmin, xmax-x])
    

    


# In[168]:


# Create data loader for training D(x,t)

d_ds_tr=Dataset_label(X_D_tr, d_tr)
d_ds_val=Dataset_label(X_D_val, d_val)

d_loader_tr=DataLoader(d_ds_tr,batch_size = len(d_ds_tr),shuffle=False)
d_loader_val=DataLoader(d_ds_val,batch_size = len(d_ds_val),shuffle=False)


# In[169]:


# create model for D

D_NN=D_NN_constr()


# In[187]:


# training function with label

def train_label(net, epochs, loader_tr, loader_val):
    loss_fcn=nn.MSELoss(reduction = 'mean')
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        if epoch == 2000:
            optimizer = optim.Adam(net.parameters(), lr=5e-4)
        running_loss=0.
        optimizer.zero_grad()
        
        data_tr, label_tr = next(iter(loader_tr))
        label_predict=net(data_tr)
        loss=loss_fcn(label_predict, label_tr)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        #print(running_loss)
        
        if epoch%50==49:
            val_loss=0.
            data_val, label_val=next(iter(loader_val))
            label_predict_val=net(data_val)
            val_loss+=loss_fcn(label_predict_val, label_val).item()
            #print(f'validation loss: {val_loss:.3f}')
    print('Finished Training',running_loss)

        
        


# In[188]:


train_label(D_NN, 3000, d_loader_tr, d_loader_val)


# In[ ]:





# In[ ]:


##########################################################################################################


# In[ ]:





# In[172]:


# Learning initial and boundary conditions

# preparing data
# use 20 points from initial, 20 points from boundary for validation

X_ib_tr=torch.cat([X_0[20:, :], X_b[20:, :]],axis=0)

# creating label

u_ib_tr=torch.cat([u_0[20:,:], u_b[20:,:]],axis=0)

    
    
# Create validation data
X_ib_val=torch.cat([X_0[:20, :], X_b[:20, :]],axis=0)

u_ib_val=torch.cat([u_0[:20,:], u_b[:20,:]],axis=0)


"""
### New idea

to prevent learning structured crap in the bulk
we also include some collocation points in the bulk
with RANDOM label between 0.5 and -0.5

But validation data are stilled just the initial and boundary points


X_ib_co=X_r[:200, :]
u_ib_co=torch.rand((200,1),dtype=DTYPE)
u_ib_co=u_ib_co - 0.5

X_ib_tr2=torch.cat([X_ib_tr, X_ib_co],axis=0)
u_ib_tr2=torch.cat([u_ib_tr, u_ib_co],axis=0)
"""
    

    


# In[173]:


# Create data loader for training IB

ib_ds_tr=Dataset_label(X_ib_tr, u_ib_tr)
ib_ds_val=Dataset_label(X_ib_val, u_ib_val)

ib_loader_tr=DataLoader(ib_ds_tr,batch_size = len(ib_ds_tr),shuffle=False)
ib_loader_val=DataLoader(ib_ds_val,batch_size = len(ib_ds_val),shuffle=False)


# In[174]:


# creating model for initial and boundary conditions

NN_IB=IB_constr()


# In[175]:


# training with the training function with label

train_label(NN_IB, 3000, ib_loader_tr, ib_loader_val)


# In[ ]:





# In[ ]:


##########################################################################################################


# In[ ]:





# In[176]:


# Train u_tilda

# preparing data
# only use collocation points
# use 100 collocation points as validation samples

r_ds_tr=Dataset_nolabel(X_r[100:,:])
r_ds_val=Dataset_nolabel(X_r[:100,:])

r_loader_tr=DataLoader(r_ds_tr,batch_size = len(r_ds_tr),shuffle=False)
r_loader_val=DataLoader(r_ds_val,batch_size = len(r_ds_val),shuffle=False)


# In[189]:


# creating model for u_tilda

u_tilda=u_tilda_constr()


# In[190]:


# training function WITHOUT label
# need the extra function get_r as an input to find the residual in the bulk
# u_net is the NN for u_tilda


def train_u_tilda(u_net, D_net, IB_net, epochs, loader_tr, loader_val, get_r):
    loss_fcn=nn.MSELoss(reduction = 'mean')
    #optimizer = optim.Adam(u_net.parameters(), lr=1e-3)
    optimizer = optim.Adam(u_net.parameters(), lr=5e-3)
    for epoch in range(epochs):
        if epoch == 200:
            #optimizer = optim.Adam(u_net.parameters(), lr=5e-4)
            optimizer = optim.Adam(u_net.parameters(), lr=1e-3)
        if epoch == 3000:
            #optimizer = optim.Adam(u_net.parameters(), lr=2e-4)
            optimizer = optim.Adam(u_net.parameters(), lr=5e-4)
        if epoch == 6000:
            #optimizer = optim.Adam(u_net.parameters(), lr=1e-4)
            optimizer = optim.Adam(u_net.parameters(), lr=1e-4)
        if epoch == 9000:
            optimizer = optim.Adam(u_net.parameters(), lr=5e-5)
        if epoch == 15000:
            optimizer = optim.Adam(u_net.parameters(), lr=1e-5)

            
        running_loss=0.
        
        optimizer.zero_grad()
        
        X_tr = next(iter(loader_tr))
        r = get_r(u_net, D_net, IB_net, X_tr)
        
        loss = loss_fcn(r,torch.zeros(r.shape))
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        print(f'epoch: {epoch}')
        print(running_loss)
        
        if epoch%50==49:
            val_loss=0.0
            X_val = next(iter(loader_val))
            r_val=get_r(u_net, D_net, IB_net, X_val)
            val_loss+= loss_fcn(r_val,torch.zeros(r_val.shape)).item()
            print(f'epoch: {epoch}')
            print(f'validation loss: {val_loss:.3f}')
    print('Finished Training',running_loss)
            
         


# In[ ]:


train_u_tilda(u_tilda, D_NN, NN_IB, 30000, r_loader_tr, r_loader_val, get_r)


# In[180]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up meshgrid
N = 600
tspace = np.linspace(lb[0], ub[0], N + 1)
xspace = np.linspace(lb[1], ub[1], N + 1)
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(),X.flatten()]).T
Xgrid = torch.tensor(Xgrid,dtype=torch.float32)


# Determine predictions of u(t, x)

upred = NN_IB(Xgrid) + D_NN(Xgrid) * u_tilda(Xgrid)
#upred = NN_IB(Xgrid)*torch.exp(-15.*D_NN(Xgrid)) + D_NN(Xgrid) * u_tilda(Xgrid)
#upred = NN_IB(Xgrid)
#upred = D_NN(Xgrid)


# Reshape upred
U = upred.detach().numpy().reshape(N+1,N+1)

# Surface plot of solution u(t,x)
fig = plt.figure(figsize=(9,6))
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
#ax.plot_surface(T, X, U, cmap='viridis');
ax.contourf(T,X,U)
#ax.view_init(90,0)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
#ax.set_zlabel('$u_\\theta(t,x)$')
ax.set_title('Solution of Burgers equation');
#plt.savefig('Burgers_solution.pdf', bbox_inches='tight', dpi=300);


# In[181]:


plt.figure()
plt.contour(X,U,T)
#plt.contour(X,U,T,levels=1)
plt.show()


# In[183]:


plt.plot(X[:, 450],U[:,450])
plt.show()


# In[184]:


plt.plot(X[:, 300],U[:,300])
plt.show()



plt.plot(X[:, 150],U[:,150])
plt.show()
# In[185]:


plt.plot(X[:, 0],U[:,0])
plt.show()


# In[56]:




# In[ ]:




