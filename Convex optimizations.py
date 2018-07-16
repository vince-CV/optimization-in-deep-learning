# -*- coding: utf-8 -*-

"""
Created on Wed Mar 28 18:01:15 2018

@author: Xunzhe Wen
"""


import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3-2*x-10+x**2  
  
def derivative_f(x):
    return 3*(x**2)+2*x-2  
  
def GD(x, learning_rate):
    gradient=0.0
    trace=[]
    
    for i in range(1000000):  
  
        if((abs(gradient)>0.00001) and (abs(gradient)<0.0001)):  
            print("Gradient Descent converges with iterations "+str(i))  
            break  
            
        else:  
            gradient = derivative_f(x)  
            x = x -  learning_rate*gradient
            trace.append(x)
            y = f(x)
            
    return x,y,trace

def Adagrad(x, learning_rate):
    gradient=0  
    e=0.00000001  
    sum = 0.0
    trace=[]
  
    for i in range(100000):  
        if((abs(gradient)>0.00001) and (abs(gradient)<0.0001)):
            print("Adagrad converges with iterations "+str(i))  
            break  
        else:  
            gradient = derivative_f(x)  
            sum += gradient**2;  
            x=x-learning_rate*gradient/(math.sqrt(sum/(i+1))+e) 
            trace.append(x)
            y=f(x)
    return x,y,trace

def RMSProp(x, learning_rate):
    gradient=0  
    e=0.00000001  
    sum = 0.0  
    d = 0.9 
    Egt=0
    Edt = 0
    delta = 0  
    trace=[]
    for i in range(100000):  
        
        if(abs(gradient)>0.00001 and (abs(gradient)<0.0001)):  
            print("RMSProp converges with iterations "+str(i))  
            break  
        else:
            gradient = derivative_f(x)  
            Egt = d * Egt + (1-d)*(gradient**2)  
            x=x-learning_rate*gradient/math.sqrt(Egt + e)  
            trace.append(x)
            y=f(x)  
    return x,y,trace

def Adam(x,learning_rate):
    gradient=0
    e=0.00000001
    b1 = 0.9  
    b2 = 0.995  
    trace=[]
  
    m = 0  
    v = 0  
    t = 0  
  
    for i in range(10000):
        #print('x = {:6f}, f(x) = {:6f},gradient={:6f}'.format(x,y,gradient))  
        if(abs(gradient)>0.00001 and (abs(gradient)<0.0001)):  
            print("Adam converges with iterations "+str(i))  
            break  
        else:
            gradient = derivative_f(x)  
  
            t=t+1  
            m = b1*m + (1-b1)*gradient  
            v = b2*v +(1-b2)*(gradient**2)
            mt = m/(1-(b1**t))  
            vt = v/(1-(b2**t))
            x = x- learning_rate * mt/(math.sqrt(vt)+e) 
            trace.append(x)
            y=f(x)  
    return x,y,trace

ini=-1.2
x1,y1,trace1=GD(ini,0.1)
x2,y2,trace2=Adagrad(ini,0.1)
x3,y3,trace3=RMSProp(ini,0.1)
x4,y4,trace4=Adam(ini,0.1)

t1=np.array(trace1)[::]
t2=np.array(trace2)[::]
t3=np.array(trace3)[::]
t4=np.array(trace4)[::]

plt.figure(figsize=(12,6))
x=np.linspace(-2,2,100000)
plt.text(-2, -3, r'$y=x^3+x^2-2x-10$',fontsize=15)
plt.plot(x,f(x)) 
plt.plot(t1,f(t1),'r*')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Optimization (3057 iteration until convergence)')
plt.grid()
plt.savefig("7.jpg") 
plt.show()

plt.figure(figsize=(12,6))
x=np.linspace(-2,2,100000)
plt.text(-2, -3, r'$y=x^3+x^2-2x-10$',fontsize=15)
plt.plot(x,f(x))
plt.plot(t2,f(t2),'r*')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Adagrad Optimization (3639 iteration until convergence)')
plt.grid()
plt.savefig("8.jpg") 
plt.show()

plt.figure(figsize=(12,6))
x=np.linspace(-2,2,100000)
plt.text(-2, -3, r'$y=x^3+x^2-2x-10$',fontsize=15)
plt.plot(x,f(x))
plt.plot(t3,f(t3),'r*')
plt.xlabel('x')
plt.ylabel('y')
plt.title('RMSProp Optimization (1774 iteration until convergence)')
plt.grid()
plt.savefig("9.jpg") 
plt.show()

plt.figure(figsize=(12,6))
x=np.linspace(-2,2,100000)
plt.text(-2, -3, r'$y=x^3+x^2-2x-10$',fontsize=15)
plt.plot(x,f(x))
plt.plot(t4,f(t4),'r*')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Adam Optimization (2253 iteration until convergence)')
plt.grid()
plt.savefig("10.jpg") 
plt.show()

print(x1,y1)
print(x2,y2)
print(x3,y3)
print(x4,y4)

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats

def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

def jacobian(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),200*(x[1]-x[0]**2)])

def hessian(x):
    return np.array([[-400*(x[1]-3*x[0]**2)+2,-400*x[0]],[-400*x[0],200]])

def gradient_descent(alpha, x0, ep, max_iter):
    
    W=np.zeros((2,max_iter))
    W[:,0] = x0
    i = 1     
    x = x0
    Report=[]
    grad = jacobian(x)
    delta = sum(grad**2)

    while i<max_iter and delta>ep:
        p = -jacobian(x)
        #x0=x
        x = x + alpha*p
        W[:,i] = x
        grad = jacobian(x)
        delta = sum(grad**2)
        Report.append(delta)
        i=i+1
        
    print('Converged using GD, iterations: ', i)
    print('Approximated Optimum:', x)

    W=W[:,0:i] 
    return x, Report, W

def adam(alpha, x, ep, max_iter):

    converged = False
    iter = 1
    W=np.zeros((2,max_iter))
    W[:,0] = x
    
    P = jacobian(x)
    J = sum(P**2)
    e=0.00000001 
    b1=0.9
    b2=0.995
    Report=[]
    
    m=0
    v=0
    t=0

    
    while not converged:
        
        grad = jacobian(x)
        
        t += 1
        
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*(grad**2)  
   
        mt = m/(1-(b1**t))  
        vt = v/(1-(b2**t))  
  
        x = x - alpha * mt/(np.sqrt(vt)+e) 
        
        W[:,iter] = x
    
        J=sum(jacobian(x)**2)
        Report.append(J)
        
        if abs(J) <= ep:
            print('Converged using Adam, iterations: ', iter)
            print('Approximated Optimum:', x)
            converged = True
        
        iter += 1 
        
        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True
    
    W=W[:,0:iter] 
    return x, Report, W

def newton(alpha, x0, ep):

    W=np.zeros((2,10**3))
    i = 1
    imax = 1000
    W[:,0] = x0 
    x = x0
    delta = 1
    Report=[]

    while i<imax and delta>ep:
        p = -np.dot(np.linalg.inv(hessian(x)),jacobian(x))
        x0 = x
        x = x + alpha*p
        W[:,i] = x
        delta = sum((x-x0)**2)
        Report.append(delta)
        i=i+1
    print('Converged using Newton, Iterations:', i)
    print('Approximated Optimum:', x)
    W=W[:,0:i] 
    return x,Report, W

#a=np.array([-0.5,2])
a=np.array([-1.2,-1])
x11,r1,w1=gradient_descent(0.001, a, 0.001, 20000)
x22,r2,w2=adam(0.1, a, 0.001, 20000)
x33,r3,w3=newton(0.1, a, 0.00001)

X1=np.arange(-1.5,1.5+0.05, 0.05)
X2=np.arange(-3.5, 5 +0.05, 0.05)
[x1,x2]=np.meshgrid(X1,X2)

f=100*(x2-x1**2)**2+(1-x1)**2

plt.figure(figsize=(8,6))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Optimazition using Steepest Descent (6585 iterations)')
plt.contour(x1,x2,f,30) 
plt.plot(w1[0,:],w1[1,:],'b*',w1[0,:],w1[1,:],'r') 
plt.savefig("3.jpg")  
plt.show()

plt.figure(figsize=(8,6))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Optimazition using Adam (965 iterations)')
plt.contour(x1,x2,f,30) 
plt.plot(w2[0,:],w2[1,:],'b*',w2[0,:],w2[1,:],'r') 
plt.savefig("4.jpg")  
plt.show()

plt.figure(figsize=(8,6))
plt.contour(x1,x2,f,30) 
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Optimazition using Newton (146 iterations)')
plt.plot(w3[0,:],w3[1,:],'b*',w3[0,:],w3[1,:],'r') 
plt.savefig("5.jpg")  
plt.show()

plt.figure(figsize=(8,6))
plt.contour(x1,x2,f,30) 
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Optimazition on rosenbrock')
plt.plot(w1[0,:],w1[1,:],'r',label='Steepest Descent') 
plt.plot(w2[0,:],w2[1,:],'g',label='Adam') 
plt.plot(w3[0,:],w3[1,:],'b',label='Newton') 
plt.legend()
plt.savefig("6.jpg")  
plt.show()

def gradient_descent(alpha, x0, ep, max_iter):
    
    W=np.zeros((2,max_iter))
    W[:,0] = x0
    i = 1     
    x = x0
    Report=[]
    grad = jacobian(x)
    delta = sum(grad**2)

    while i<max_iter and delta>ep:
        p = -jacobian(x)
        #x0=x
        x = x + alpha*p
        W[:,i] = x
        grad = jacobian(x)
        delta = sum(grad**2)
        Report.append(delta)
        i=i+1
        
    print('Converged using GD, iterations: ', i)
    print('Approximated Optimum:', x)

    W=W[:,0:i] 
    return x, Report, W

def adam(alpha, x, ep, max_iter):

    converged = False
    iter = 1
    W=np.zeros((2,max_iter))
    W[:,0] = x
    
    P = jacobian(x)
    J = sum(P**2)
    e=0.00000001 
    b1=0.9
    b2=0.995
    Report=[]
    
    m=0
    v=0
    t=0

    
    while not converged:
        
        grad = jacobian(x)
        
        t += 1
        
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*(grad**2)  
   
        mt = m/(1-(b1**t))  
        vt = v/(1-(b2**t))  
  
        x = x - alpha * mt/(np.sqrt(vt)+e) 
        
        W[:,iter] = x
    
        J=sum(jacobian(x)**2)
        Report.append(J)
        
        if abs(J) <= ep:
            print('Converged using Adam, iterations: ', iter)
            print('Approximated Optimum:', x)
            converged = True
        
        iter += 1 
        
        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True
    
    W=W[:,0:iter] 
    return x, Report, W

new=np.array([-0.5,2])
#new=np.array([-1.2,-1])

x_adam,r2,w22=adam(0.1, new, 0.001, 900)
x_com,r1,w11=gradient_descent(0.001, x_adam, 0.001, 20000)

X1=np.arange(-1.5,1.5+0.05, 0.05)
X2=np.arange(-3.5, 5 +0.05, 0.05)
[x1,x2]=np.meshgrid(X1,X2)

f=100*(x2-x1**2)**2+(1-x1)**2

plt.figure(figsize=(8,6))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Optimazition using Steepest Descent + Adam')
plt.contour(x1,x2,f,30) 
plt.plot(w22[0,:],w22[1,:],'b',w11[0,:],w11[1,:],'r') 
plt.savefig("3.jpg")  
plt.show()
