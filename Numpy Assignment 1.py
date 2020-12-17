#!/usr/bin/env python
# coding: utf-8

# # Loading Data from Onlion Source

# In[1]:


import numpy as np
import urllib.request
urllib.request.urlretrieve('https://courses.washington.edu/b517/Datasets/shoulder.csv','Data_SET_From_onlion_Source.txt')
fatched_Data=np.genfromtxt('Data_SET_From_onlion_Source.txt',delimiter=',',skip_header=1)


# # Use of whare on file Data

# In[2]:


fatched_Data.shape
array_refilling=np.where(fatched_Data==0,"zerso_value",fatched_Data)
array_refilling


# 
# # Statical and mathmatical Function || Attributes

# In[3]:


mean=fatched_Data.mean()
variance=fatched_Data.var()
standard_deviation=fatched_Data.std()
min_value=fatched_Data.min()
max_value=fatched_Data.max()
product_of_matrix=np.average(fatched_Data)
percentile_of_Data=np.percentile(fatched_Data,100)
maxandmin=np.ptp(fatched_Data,0)
print('mean_of_file{}\nvarience of file:{}\nstanderd daviation:{}\nmin value :{}\nmax value:{}\n average of matrix :{} \npercintile of data :{}\n max and min of array:{}\n'.format(mean,variance,standard_deviation,min_value,max_value,product_of_matrix,percentile_of_Data,maxandmin))


# # File Operations

# In[4]:


fatched_Data
np.save('Demo_File',fatched_Data)
np.load('Demo_File.npy')


# In[5]:


array1=np.array([1,2,3,4,5])
array2=np.array([6,7,8,9,10])
np.savez('Save_2_Array',x=array2,y=array1)
a=np.load('Save_2_Array.npz')
a['x']


# # Linear Algebric Functions

# In[6]:


Trace=fatched_Data.trace()
Transpose=fatched_Data.transpose()
A = np.array([[6, 1, 1],
              [4, -2, 5],
              [2, 8, 7]])
np.linalg.inv(A)
np.linalg.matrix_rank(A)
np.linalg.det(A)
np.linalg.matrix_power(A,2)
np.linalg.eig(A)


# # Array Manipulation

# In[7]:


# Reshape
A=np.random.randn(4,4)
B=A.reshape(4,2,2)
B.dtype
B.shape
A.reshape((4,2,2),order='f')
#converting any array shape into single dim
A.flatten()
A.ravel()
#concatination
A = np.array([[6, 1, 1],
              [4, -2, 5],
              [2, 8, 7]])   
b =np.array([[1,1,1],
             [1,1,1],
             [1,1,1]])
np.concatenate((A,b),axis=1)
vstackimp=np.array([[6, 1, 1],
                   [4, -2, 5],
                   [2, 8, 7]]) 
data =np.array([[1,1,1],
               [1,1,1],
               [1,1,1]])
np.hstack((vstackimp,data))
#spliting
vstackimp=np.array([6, 1, 1,2,2,2,2]) 
np.split(vstackimp,(2,5))
np.hsplit(vstackimp,[1])
np.split(data,[1,3])
test_split=np.array([[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,16,17]])
np.split(test_split,2)
np.repeat(test_split,2)
np.tile(test_split,2)


# # Gernating Arrays and Changing Data type

# In[8]:


np.zeros(4)
cHECK_ONE_like=np.ones(100,dtype=float).reshape(50,2)
np.identity(3)
np.eye(12)
np.empty(122)
#now i will wroye like one function that make same number data with given var.
np.ones_like(fatched_Data)
#data data_type changing
cHECK_ONE_like.dtype
a=cHECK_ONE_like.astype(np.int)
a.dtype
#we can aslo set dtype during makig ndarray
array=np.array([1,2,3,4,5,6,7,8,9,0],dtype=str)
array.dtype


# # Arthmatic Operation

# In[9]:


#hare web will discuss about arthmatic how we can sum mula and div without loop just by a little function
arthmatic_operation_testing_arrayA=np.array([[1,1,1,1],[1,1,1,1]])
arthmatic_operation_testing_arrayB=np.array([[1,1,1,1],[1,1,1,1]])
new_mul_array=arthmatic_operation_testing_arrayA*2
new_mul_array*arthmatic_operation_testing_arrayB


# # Pseudorandom Number Generation

# In[20]:


np.random.seed(11)
samples = np.random.normal(size=(4, 4))
samples


# In[26]:


arr1 = [[1, 2], 
        [3, 4.]]
arr2 = [[5, 6, 7], 
        [8, 9, 10]]
np.concatenate((arr1, arr2), axis=1)
arr1 = [["apple", "manago"],
        ["banana", "watermelon"]]
arr2 = [["dog", "lion"],
        ["cat", "tiger"]]
np.char.add(arr1, arr2)
a = np.arange(16).reshape(2,2,4) 

print ('The original array is:' )
print (a)  
print ('\n') 

print('The transposed array is:') 
print (np.transpose(a,(1,0,2)))
arr2=(0,0,4,0,0,5,6,0,0)
np.trim_zeros(arr2)
arr1=np.array([1,2,3,4,3,2,1,4])
np.unique(arr1)


# In[ ]:




