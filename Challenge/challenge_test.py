# Run this cell to import the packages you will need to unpack the dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy
from io import BytesIO
from PIL import Image
import random
import pickle
import os
import zipfile
import scipy.ndimage
#from google.colab import files
from skimage import feature
import tensorflow as tf

os.chdir('Data')                               # Don't rerun these two lines!


train_simple_labels = pd.read_csv('train_simple_labels.csv', header = None)
train_complex_labels = pd.read_csv('train_complex_labels.csv', header=None)
eval_simple_labels = pd.read_csv('eval_simple_labels.csv', header=None)
eval_complex_labels = pd.read_csv('eval_complex_labels.csv', header=None)

train_simple_labels = pd.concat([train_simple_labels, eval_simple_labels], axis=0, ignore_index = True)
train_complex_labels = pd.concat([train_complex_labels, eval_complex_labels], axis=0, ignore_index = True)
Labels = pd.DataFrame(list(range(5500)))
train_simple_labels = pd.concat([train_simple_labels, Labels], axis=1)
train_simple_labels.columns = ['Label', 'Unique_Index']
train_complex_labels = pd.concat([train_complex_labels, Labels], axis=1)
train_complex_labels.columns = ['Label', 'Unique_Index']



zip_ref = zipfile.ZipFile('Mamm_Images_Train.zip.zip', 'r')
zip_ref.extractall('Mamm_Images_Train')
zip_ref.close()

zip_ref = zipfile.ZipFile('Mamm_Images_Eval_zip.zip', 'r')
zip_ref.extractall('Mamm_Images_Eval')
zip_ref.close()

zip_ref = zipfile.ZipFile('Mamm_Images_Test.zip.zip', 'r')
zip_ref.extractall('Mamm_Images_Test')
zip_ref.close()  


train_images = []
for i in range(5000):
  im = scipy.ndimage.imread('Mamm_Images_Train/Mamm_Images_Train/image' + str(i) + '.jpg')
  train_images.append(im)
  
for h in range(500):
  im = scipy.ndimage.imread('Mamm_Images_Eval/Mamm_Images_Eval/image' +str(h) + '.jpg')
  train_images.append(im)
  
train_images_df = pd.DataFrame([train_images])
train_images_df = train_images_df.transpose()
Labels = pd.DataFrame(list(range(5500)))
train_images_df = pd.concat([train_images_df, Labels], axis = 1)
train_images_df.columns = ['Images', 'Unique Index']


test_images = []
for k in range(1500):
  im = scipy.ndimage.imread('Mamm_Images_Test/Mamm_Images_Test/image' + str(k) + '.jpg')
  test_images.append(im)
  
  
test_images_df = pd.DataFrame([test_images])
test_images_df = test_images_df.transpose()
Labels = pd.DataFrame(list(range(5500, 7000)))
test_images_df = pd.concat([test_images_df, Labels], axis = 1)
test_images_df.columns = ['Images', 'Unique Index']

train_labels = train_simple_labels['Label'].as_matrix()
train_images = np.array(train_images)
train_images = train_images.reshape((train_images.shape[0],train_images.shape[1],train_images.shape[2],1))

from tensorflow.python.keras.layers import Dense, Conv1D, Conv2D, Flatten, Dropout, MaxPool1D, MaxPool2D
from tensorflow.python.keras.models import Sequential

model = Sequential()

model.add(Conv2D(filters=16,kernel_size=(3,3),strides=(3,3),input_shape=(train_images.shape[1],train_images.shape[2],1),activation='relu'))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='sigmoid'))
model.add(Dense(1,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,
          train_labels,
          batch_size=100,
          epochs=20,
          validation_split=0.2)
