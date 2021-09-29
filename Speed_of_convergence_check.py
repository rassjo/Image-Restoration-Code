# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 13:13:56 2021

@author: joyce
"""

# Import libraries
from PIL import Image as im
import numpy as np
from scipy import stats
from scipy.stats import chisquare
import matplotlib.pyplot as plt

# Choose the image to reconstruct
#path = 'Geometric figures/'
#path = 'Geometric figures/'
#path = 'Hand writing/'
path = 'Text/'

# Open the original image, the grafiti image and the mask image
# Convert them to greyscale using .convert('L')
Im1=im.open(path+'Original.jpg')
original=Im1.convert('L')
Im2=im.open(path+'Graffiti.jpg')
graffiti=Im2.convert('L')
# The mask is bigger than the actual grafiti to avoid problems at the boundary (Isabel's findings)
Im3=im.open(path+'Mask.jpg')
mask=Im3.convert('L')

# Restore the image from an array
def restored_img(n, m, inside_array, g_array, u):
    rs = g_array
    for i in range(0,n):
        for j in range(0,m):
            if inside_array[i,j] >= 1:
                rs[i,j]=u[i,j]
    return rs

# Calculate the Chi Square 
def chi_squared(rs, o_array, inside_array, n, m):
    i = n
    j = m
    rs_test = []
    o_array_test = []
    sigma_array = []
    chi_array = []
    # create two arrays, one containing the original image values and another containing the inpainted pixel values
    for a in range (0,i):
        for b in range (0,j):
            if inside_array[a,b] == 1:
                rs_test.append(rs[a,b])
                o_array_test.append(o_array[a,b])
    mean = np.mean(o_array_test)
    # Compute sum for sigma
    for x in range(0, len(o_array_test)):
        element = (o_array_test[x] - mean)**2
        sigma_array.append(element)
    sigma_squared = (1 / (len(rs_test)-1))*np.sum(sigma_array)
    print(sigma_squared)
    # Compute sum in nominator
    for x in range (0, len(rs_test)):
        value = (rs_test[x]-float(o_array_test[x]))**2
        chi_array.append(value)
    # Print statement to test if chi square behaves as expected
    print(np.sum(chi_array))
    chi = (1/len(chi_array))*((np.sum(chi_array))/sigma_squared)
    rs_test = []
    o_array_test = []
    return chi

# Converting images to arrays
o_array = np.array(original)
g_array = np.array(graffiti)
m_array = np.array(mask)
t_array = g_array

# Define the shape of the image
shape = g_array.shape
n = int(shape[0])
m = int(shape[1])

# Initializing empty arrays with the shape of the picture.
# boundary_array contains boundary conditions (not used anymore!)
# indside_array defines which pixels are inside the mask
boundary_array = np.zeros((n,m))
inside_array = np.zeros((n,m))
u0 = np.zeros((n,m))

shape = g_array.shape
n = shape[0]
m = shape[1]

for i in range (1, (n-1)):
    for j in range (1, (m-1)):
        # Create a matrix where the elements corresponding to inside have value 1
        if m_array[i,j]>210:
            m_array[i,j]=255
            inside_array[i,j]=1
        else:
            # Create a matrix where the elements corresponding to boundaries have value 2
            m_array[i,j] = 0
            if m_array[i-1][j]>210 or m_array[i+1][j]>210 or m_array[i][j-1]>210 or m_array[i][j+1]>210:
                inside_array[i,j]= 2

# Create a 2D array that contains the original values inside the mask area
for i in range (0,n):
    for j in range(0,m):
        # Fill u only with the values of coordinates corresponding to the area to be restored 
        if inside_array[i,j] == 1:
            u0[i,j] = float(g_array[i,j])
        if inside_array[i,j] == 2:
           u0[i,j] = float(g_array[i,j])

# Define the constants to reconstruct the image
# 'Count' defines the number of iterations
# 0 < w < 2 depending on whether one wants Gauss-Seidel (w=1), Underrelaxation (0 < w < 1) or Overrelaxation (1 < w < 2)
count = 45
u = u0     
w = 0.5
chi_2_array_0 = [] 
n_array = []
counter = 0
    
# We loop over array u to reconstruct the images      
while count > 0:
    for i in range(0,n):
        for j in range(0,m):
            if inside_array[i,j] == 1 :
                u[i,j] = (1-w)*u[i,j] + (w/4)*(u[(i+1), j] + u[(i-1), j] + u[i, (j+1)] + u[(i, (j-1))])
    # Compute the chi squared value at every nth step 
    if count % 1 == 0:
        rs = restored_img(n, m, inside_array, g_array, u) 
        chi_2 = chi_squared(rs, o_array, inside_array, n, m)
        chi_2_array_0.append(chi_2)
        n_array.append(counter)
        print(chi_2)
    counter +=1
    count-=1

# Save the chi square values to compute the graphs
np.save(('Text_w'+str(w)), chi_2_array_0) 


rs = g_array
for i in range(0,n):
    for j in range(0,m):
        if inside_array[i,j] ==1:
            rs[i,j]=u[i,j]
restored=im.fromarray(rs)
restored.show()

''' 
# Compute the graphs
chi_0 = np.load('Text_w1.5.npy')
chi_1 = np.load('Text_w1.0.npy')
chi_2 = np.load('Text_w0.5.npy')

plt.plot(n_array, chi_0, label="Overrelaxation (w = 1.5)")
plt.plot(n_array, chi_1, label="Gauss-Seidel (w = 1.0)")
plt.plot(n_array, chi_2, label="Underrelaxation (w = 0.5)")
plt.xlabel("Number of iterations")
plt.ylabel("Chi^2")
plt.title("Convergence of 'Text'")
plt.legend()
plt.show()
'''


            
                

