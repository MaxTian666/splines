from scipy.interpolate import interp1d
from scipy.interpolate import griddata

import numpy as np 
import matplotlib.pyplot as plt 


x = np.linspace(0,10,num=11,endpoint=True)
y = np.cos(-x**2/9.0)

f = interp1d(x,y)
f2 = interp1d(x,y,kind='cubic')       #1维插值函数，‘cubic’为3次插值

xnew = np.linspace(0,10,num=41,endpoint=True)

plt.plot(x,y,'o',xnew,f(xnew),'*',xnew,f2(xnew),'.')



def func(x,y):
    return x*(1-x)*np.cos(4*np.pi*x)*np.sin(4*np.pi*y**2)**2

grid_x,grid_y = np.mgrid[0:1:100j,0:1:200j]
points = np.random.rand(1000,2)
values = func(points[:,0],points[:,1])

grid_z0 = griddata(points,values,(grid_x,grid_y),method='nearest')
grid_z1 = griddata(points,values,(grid_x,grid_y),method='linear')
grid_z2 = griddata(points,values,(grid_x,grid_y),method='cubic')

plt.figure(2)
plt.subplot(221)
plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0], points[:,1], 'k.', ms=1)
plt.title('Original')

plt.subplot(222)
plt.imshow(grid_z0.T,extent=(0,1,0,1),origin='lower')
plt.title('Nearest')

plt.subplot(223)
plt.imshow(grid_z1.T,extent=(0,1,0,1),origin='lower')
plt.title('linear')

plt.subplot(224)
plt.imshow(grid_z2.T,extent=(0,1,0,1),origin='lower')
plt.title('Cubic')

plt.gcf().set_size_inches(6,6)


#三次样条插值
from scipy import interpolate
x = np.arange(0,2*np.pi+np.pi/4,2*np.pi/8)
y = np.sin(x)

tck = interpolate.splrep(x,y,s=0)
xnew = np.arange(0,2*np.pi,np.pi/50)
ynew = interpolate.splev(xnew,tck,der=0)

plt.figure(3)
plt.subplot(311)
plt.plot(x,y,'x',xnew,ynew,xnew,np.sin(xnew),x,y,'b')
plt.legend(['Linear','Cubic Spline','True'])
plt.axis([-0.05,6.33,-1.05,1.05])
plt.title("Cubic-spline interpolation")


#样条求导
yder = interpolate.splev(xnew,tck,der=1)
plt.subplot(312)
plt.plot(xnew,yder,xnew,np.cos(xnew),'*')
plt.legend(['Cubic Spline','True'])
plt.axis([-0.05,6.33,-1.05,1.05])
plt.title("Derivative estimation from spline")

#样条积分
def integ(x,tck,constant=-1):
    x = np.atleast_1d(x)
    out = np.zeros(x.shape,dtype=x.dtype)
    for n in range(len(out)):
        out[n] = interpolate.splint(0,x[n],tck)   #求取任意两点间的积分
    out += constant
    return out


yint = integ(xnew,tck)
plt.subplot(313)
plt.plot(xnew,yint,xnew,-np.cos(xnew),'--')
plt.legend('Cubic Spline','True')
plt.axis([-0.05,6.33,-1.05,1.05])
plt.title("Integral estimation from spline")

plt.show()


