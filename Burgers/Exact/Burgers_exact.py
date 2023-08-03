from scipy.integrate import quad,quadrature
import numpy as np
import matplotlib.pyplot as plt

nu=0.01/np.pi

# First do denominator

def ker(y,t):
    return np.exp(- y*y / (4*nu*t))

def f(y,x,t):
    return np.exp(-np.cos(np.pi*(x-y))/(2*np.pi*nu))

def integrand_d(y,x,t):
    return f(y,x,t)*ker(y,t)

def denominator(x,t):
    return quad(integrand_d, -np.inf, np.inf, args=(x,t))[0]
    #return quadrature(integrand_d, -10., 10., args=(x,t))[0]
# Now the numerator

def sin(y,x):
    return np.sin(np.pi*(x-y))

def integrand_n(y,x,t):
    return sin(y,x)*f(y,x,t)*ker(y,t)

def numerator(x,t):
    return quad(integrand_n, -np.inf, np.inf, args=(x,t))[0]
    #return quadrature(integrand_n, -10., 10., args=(x,t))[0]


def u(x,t):
    return -numerator(x,t)/denominator(x,t)
    

# for t=0.25, 0.5, 0.75, store 201 evenly spaced point from -1 to 1

u_list=np.zeros((201,3))
u_list_1=[]
u_list_2=[]
u_list_3=[]

for x in np.arange(-1,1.01,0.01):
    u_list_1.append(u(x,0.25))
    u_list_2.append(u(x,0.5))
    u_list_3.append(u(x,0.75))

u_list[:,0]=np.array(u_list_1)
u_list[:,1]=np.array(u_list_2)
u_list[:,2]=np.array(u_list_3)


plt.plot(np.arange(-1,1.01,0.01), u_list_2)








