import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import odeint


def fx(x,y,z,t): return sigma*(y-x)
def fy(x,y,z,t): return x*(rho-z)-y
def fz(x,y,z,t): return x*y-beta*z

#a) Defining the classical Runge-Kutta 4th order method

def RungeKutta45(x,y,z,fx,fy,fz,t,h):
    k1x,k1y,k1z = ( h*f(x,y,z,t) for f in (fx,fy,fz) )
    xs, ys,zs,ts = ( r+0.5*kr for r,kr in zip((x,y,z,t),(k1x,k1y,k1z,h)) )
    k2x,k2y,k2z = ( h*f(xs,ys,zs,ts) for f in (fx,fy,fz) )
    xs, ys,zs,ts = ( r+0.5*kr for r,kr in zip((x,y,z,t),(k2x,k2y,k2z,h)) )
    k3x,k3y,k3z = ( h*f(xs,ys,zs,ts) for f in (fx,fy,fz) )
    xs, ys,zs,ts = ( r+kr for r,kr in zip((x,y,z,t),(k3x,k3y,k3z,h)) )
    k4x,k4y,k4z  =( h*f(xs,ys,zs,ts) for f in (fx,fy,fz) )
    return (r+(k1r+2*k2r+2*k3r+k4r)/6 for r,k1r,k2r,k3r,k4r in 
            zip((x,y,z),(k1x,k1y,k1z),(k2x,k2y,k2z),(k3x,k3y,k3z),(k4x,k4y,k4z)))


sigma=10.
beta=8./3.
rho=28.
tIn=0.
tFin=40.
h=0.01
totalSteps=int(np.floor((tFin-tIn)/h))

t = totalSteps * [0.0]
x = totalSteps * [0.0]
y = totalSteps * [0.0]
z = totalSteps * [0.0]

x[0],y[0],z[0],t[0] = 1., 1., 1., 0.  #Initial condition
for i in range(1, totalSteps):
    x[i],y[i],z[i] = RungeKutta45(x[i-1],y[i-1],z[i-1], fx,fy,fz, t[i-1], h)


#Plotting solution

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot(x,y,z,'r',label='Lorentz 3D Solution', color = 'green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()