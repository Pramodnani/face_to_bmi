#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import os
import sys


# In[2]:


import glob  #file pattern matching


# In[3]:


print(os.getcwd())


# #to resize all the images to 150*150 size
# width=150
# height=150
# img_num=0
# for filename in glob.glob('Images/*.bmp'): #the pattern matches every pathname inthe directory test_images 
#     im=Image.open(filename).convert('L')#pillow to load image and convert to greyscale image
#     out=im.resize((width,height),Image.ANTIALIAS)
#     out.save("C:/Users/pramo/Downloads/face-to-bmi/Data/Resize_images/img_{}.bmp".format(img_num))
#     img_num+=1
#    

# In[4]:


image_list=[]
for filename in glob.glob('Resize_images/*.bmp'): #the pattern matches every pathname inthe directory test_images 
    img=Image.open(filename)#pillow to load image
    image_list.append(img)#adding images to image_list


# In[5]:


len(image_list)


# In[6]:


image_list[67]


# In[11]:


#to convert images to array=>1d array=>2d array 
#appending all the image vectors to X_img vector
temp=1
for pic in image_list:
    arr=np.array(pic)#to convert image to array 
    arr_1d=arr.flatten()#to convert into 1D array
    arr_2d=arr_1d.reshape(1,22500) #to convert 1D to 2D array
    if(temp==1):
        X_img=arr_2d
        temp=0
    else:
        X_img=np.append(X_img,arr_2d,axis=0)#to add new rows to the X_image data


# In[12]:


X_img.shape


# In[13]:


#standardising data(setting mean=0,variance=1)
from sklearn.preprocessing import StandardScaler


# In[14]:


X_P_img=StandardScaler().fit_transform(X_img)


# In[15]:



from sklearn.decomposition import PCA


# In[16]:


#princinple components choosen should be 95% of variance
pca=PCA(.95)


# In[17]:


#to find out principle components from X_P_img
pca.fit(X_P_img)


# In[18]:


#no:of principle components
pca.n_components_


# In[19]:


X_P_img=pca.transform(X_P_img)


# In[20]:


X_P_img.shape


# In[21]:



import pandas as pd
#to read_csv file
data_unclean=pd.read_csv('data.csv')
#to show top 5rows
data_unclean.head()


# In[22]:


#to remove unnamed coloums in data
data=data_unclean.loc[:,~data_unclean.columns.str.contains('^Unnamed')]

data.head()


# In[23]:


#should include is_training?
x_train=data.loc[:,['gender']]

x_train.head()


# In[24]:


from sklearn.preprocessing import LabelEncoder#to convert categorical data to numerical data(gender col)


# In[25]:


number=LabelEncoder()


# In[26]:


x_train=number.fit_transform(x_train['gender'].astype('str'))


# In[27]:


x_train=x_train.reshape(4206,1)


# In[28]:


y_train=data.loc[:,['bmi']]
y_train=np.array(y_train)
y_train.shape


# In[29]:


X_train=np.append(x_train,X_P_img,axis=1)
X_train.shape


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)


# In[32]:


X_train.shape


# In[33]:


y_train.shape


# In[34]:


#pd.DataFrame(X_train)


# In[35]:


X_test.shape


# In[36]:


X_train.shape


# In[37]:


y_train=y_train.reshape(3364,)


# In[40]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


# gammas=[0.001,0.01,0.1,1,10]#higher gamma considers points near decision boundary
# c=[0.1,1,10,100,1000]#fitting the decision boundary line perfectly to the data points, higher values results ovefitting
# #linear,rbf,poly shape of hyperplane that separates
# param_grid={'C':c,'gamma':gammas}
# regression=SVR(kernel='rbf')
# grid_search=GridSearchCV(regression,param_grid)
# model=grid_search.fit(X_train,y_train)
#     #prediction=model.predict(X_test)
# print(model.best_params_)
#    # MSE=mean_squared_error(y_test,prediction)
#     #score=r2_score(y_test,prediction)
#     #print("C:{},mean error{},r2score{}".format(g,MSE,score))

# In[ ]:


#best parameters are c=100, gamma=0.001

regression=SVR(C=100,gamma=0.001)


# In[66]:


model=regression.fit(X_train,y_train)


# In[67]:


prediction=model.predict(X_test)


# In[68]:


MSE=mean_squared_error(y_test,prediction)
score=model.score(X_test,y_test)
print("MSE={} and r2score={}".format(MSE,score))


# In[69]:


from matplotlib import pyplot


# In[70]:


pyplot.scatter(y_test,prediction)
pyplot.xlabel('true values')
pyplot.ylabel('predicted')


# In[71]:


def export_prediction_SVR():
    return prediction


# In[72]:


def export_ytest_SVR():
    return y_test


# In[ ]:




