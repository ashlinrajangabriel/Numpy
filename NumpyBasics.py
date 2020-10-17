#Credits go to Lazy programmer check out his course. he has great stuff.
#https://www.udemy.com/course/numpy-python/learn/lecture/19425070#overview
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:58:14 2020

@author: ASHLIN GABRIEL
"""

import numpy as np
a = np.array([1,2,3])
b = np.array([1,2,3])
print("a shape is ",a.shape)

dot = 0
for e,f in zip(a,b):
    dot += e*f
dot

print(dot)

#Understand fifference between List and array
L = [1,2,3]

L + [5]
A+ np.array([4])

A + np.array([4,5,6])

2*L #DUplicates twice
2*A #Multipies

A**2 #Square
np.log(A)
np.exp(A)
np.tanh(A)

a1 = np.array([1,2])
b1 = np.array([3,4])

dot = 0
for e,f in zip(a1,b1):
    dot += e * f
dot
    
    
dot = 0
for i in range(len(a)):
    dot += a[i]*b[i]
dot
    
np.sum(a * b)    

(a*b).sum()
a.dot(b)
a @ b
#Linear algebra

 # aT b = ||a|| ||b|| cosΘ ab 
#  cosΘ ab = aT b / ||A|| ||B|| 
#magnitude of a found
# ||A||
#Norm is magnitude
amag = np.sqrt((a * a).sum())

np.linalg.norm(a)
cosangle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.arccos(cosangle)
angle


#Matrices

L = [[1,2],[3,4]]
L[0][1]
A = np.array([[1,2],[3,4]])
#A[: Row,Columns]
A[:,0]
np.exp(A)
np.exp(L)

np.linalg.det(A)

np.linalg.inv(A)

np.trace(A)

np.diag(A)

#You give vector you get matrix and vice versa

#Eigen vectors

Lam , V = np.linalg.eig(A)

V[:,0]*Lam[0] == A @ V[:,0] #Its the same but due to precision it returns false

#Eigen value would impact the determinant to bring slope to the same line and 0

V[:,0]*Lam[0] , A @ V[:,0]

np.allclose(V[:,0]*Lam[0] , A @ V[:,0])


#Lets solve linear systems

A = np.array(([1,1],[1.5,4]))
B = np.array([2200,5050])

np.linalg.solve(A,B)


#Gemnerating Data

np.zeros((2,3))

np.ones((2,3))

10 * np.ones((2,3))

#Identity matrix

np.eye(3)



#Generating random numbers

np.random.random()

np.random.random((2,3))

#Distributions

np.random.randn(2,3)

np.random.randn(1000)

np.random.randn(1000,3)

R = np.random.randn(1000,3)
R.mean(axis = 1 ).shape

np.cov(R)

np.cov(R.T)

np.cov(R, rowvar = False)

np.random.randint(0,10,size = (3,3))

