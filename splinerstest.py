from scipy.interpolate import interp1d
import numpy as np 
import matplotlib.pyplot as plt 


x = np.linspace(0,10,num=11,endpoint=True)
y = np.cos(-x**2/9.0)

f = interp1d(x,y)
f2 = interp1d(x,y,kind='cubic')       #1维插值函数，‘cubic’为3次插值

xnew = np.linspace(0,10,num=41,endpoint=True)

plt.plot(x,y,'o',xnew,f(xnew),'*',xnew,f2(xnew),'.')

plt.show()
