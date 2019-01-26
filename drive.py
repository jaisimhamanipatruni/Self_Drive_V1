
# coding: utf-8

# In[37]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import keras 
import cv2 


# In[38]:


import csv 

lines = [] 

with open('driving_log.csv') as csvfile :
    reader = csv.reader(csvfile)
    for line in reader :
        lines.append(line)


# In[39]:


images = [] 
measurements  = [] 

        
image  = cv2.imread("driving_log.csv") 
images.append(image)
measurement = float(line[3])
measurements.append(measurement) 


# In[40]:


x_train = np.array(images)
y_train = np.array(measurements)


# In[41]:


#NeuralNetwork 
from keras.models import Sequential 
from keras.layers import Dense , Flatten  
model = Sequential() 


# In[42]:


model  = Sequential() 
model.add(Flatten(input_shape =(160,320,3)))
model.add(Dense(1)) 


# In[43]:


model.compile(loss= 'mse' ,optimizer = 'adam')


# In[44]:


model.fit(x_train ,y_train , validation_split= 0.2 , shuffle = True, epochs = 7 )


# In[46]:


model.save('model.h5')

