# -*- coding: utf-8 -*-
"""
Created on Sat May 10 11:09:26 2014

@author: huajh
"""

from numpy import *

def CreateSampleData():
    
    rad = sqrt(random.rand(100,1)); # Radius
    Ang = 2*math.pi*random.rand(100,1); #Angle
    data = [];
    label = [];
    return data,label;

def linear_kernel(x,y):
    # x,y num  x dim
    return dot(x,y.T)
 
def poly_kernel(x,y):
    gamma = 0.5;
    r = 0;
    d = 3;
    return (gamma*dot(x,y.T)+r)**d
def rbf_kernel(x,y):
    sigma = 0.5;
    return exp( -(1/(2*sigma**2))* (x**2 + y.T*2 - 2*dot(x,y.T)))

def exp_kernel(x,y):
    sigma = 0.5;
    return exp( -(1/(2*sigma**2)) * abs(x - y.T) )


def trainsvm(data,label):
    label[label<0] = -1
    label[label>=0] = 1    
    ker_func = rbf_kernel;
    num,dim = data.shape;
    Q = dot(dot(label,label.T),ker_func(data,data))
    e = -ones(num,1);
    A = label.T
    b = zeros(num,1);
    C = ones(num,1);
    QP(Q,e,A,b,C);
    

def QP(Q,e,A,b,C):
    # sovle 
    # min f(x) = 1/2 x^T Q x + e^T x
    # st. Ax = b
    H  = 1
    
if __name__ == '__main__':
   # data,label = CreateSampleData();
    num = 100;
    rad1 = sqrt(random.rand(num,1)); # Radius
    ang1 = 2*math.pi*random.rand(num,1); #Angle
    data1 = zeros([num,2]);
    data1[:,0] = (rad1*cos(ang1))[:,0]
    data1[:,1] = (rad1*sin(ang1))[:,0]
    
    rad2 = sqrt(3*random.rand(num,1)+1); # Radius
    ang2 = 2*math.pi*random.rand(num,1); #Angle
    data2 = zeros([num,2]);
    data2[:,0] = (rad2*cos(ang2))[:,0]
    data2[:,1] = (rad2*sin(ang2))[:,0]
    
    fig1 = plt.figure(1)
    plt.grid(True)
    plt.plot(data1[:,0],data1[:,1],'go');
    plt.plot(data2[:,0],data2[:,1],'ro');
    
    cir = plt.Circle((0, 0), 1, facecolor='none',edgecolor='g', linewidth=2, alpha=0.5)
    plt.gca().add_patch(cir)
    cir = plt.Circle((0, 0), 2, facecolor='none',edgecolor='r', linewidth=2, alpha=0.5)
    plt.gca().add_patch(cir)    
    plt.axis('equal');             
    
    traindata = array(data1.tolist() + data2.tolist()); 
    label = label = ones([2*num,1])
    label[num:2*num] = -1
    # two class
    trainsvm(data,label);
    
    rad0 = sqrt(4*random.rand(num,1)); # Radius
    ang0 = 2*math.pi*random.rand(num,1); #Angle
    testdata = zeros([num,2]);
    testdata[:,0] = (rad0*cos(ang0))[:,0]
    testdata[:,1] = (rad0*sin(ang0))[:,0]
    plt.plot(testdata[:,0],testdata[:,1],'b*');
    
    plt.show();
    fig1.savefig("hw5_svm_data.pdf");
    
    
    