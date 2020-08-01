# here we are implementing transfer learning using vgg16 model..
# you known about transfer learning.
# this is tranfer learning for object recognize.
from keras.layers import Input, Lambda, Dense, Flatten # here we are implementing keras layers
#we are importing this beacuse , we will create last layer for our output, vgg16 model developement 10000 images classifier, so we dont need of 1000
# classifier nueron on the last layer, so we are modify this last layer according to our problem statement using flatten,dense layers
from keras.models import Model
from keras.applications.vgg16 import VGG16 # VGG16 its preimplement and predefine model. we will use this model. this model preimplement in keras liberary
# for our problem object recongize.
# use preimplemented model for another similer type  problem statements , that concept is called tranfer learning. 
from keras.applications.vgg16 import preprocess_input # it for preprocess our input..
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator # its liberary use for image arugmentaion. its help us to generate new images by zooming in zooming out, horiental flip , vertical flip etc.
from keras.models import Sequential #implement deep learning model.
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224] # this VGG16 model take image size is 224,224.

train_path = 'Datasets/Train' # its path of train dataset..
valid_path = 'Datasets/Test' # its path of test dataset..

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# here input_shape=IMAGE_size +[3] mean , our image size 224,224 *3 , its basically mean our image is RBG image.
# weights=imagenet , it already present in keras vgg16 model, so we are not modifiying it.
#include_top=false - its very important parameter , using this we are tellling last layer we are adding in vgg6 model, once last layer remove , when we will use this model for transfer leraning.

# here we dont have  train our vgg16 layers, because its all layers are already trained , weights are fixed 
for layer in vgg.layers: # here we are iterating all the layer
  layer.trainable = False # here tell them we are not train , so we give false
  
  # if you dont do this , your model will start train for your small data and give bad accuracy..
  # becoz it vgg16 trained with many many images.it trained on 10 m of images.
# dont need to train this model. becoz if you train again for good accuaracy , you required large dataset and high computation power.
  

  
  # useful for getting number of classes
folders = glob('Datasets/Train/*') # using this we check , how many catergorize basically we have. 
#this check how many folder we have of catergorize. , 
  

# our layers - you can add more if you want
x = Flatten()(vgg.output) # here we are flating last layer of vgg16. for add  last layer of my problem statement in the vgg16
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x) # this statement add categorize in last layer of x. using activation function softmax. using this statement , our mannual create output layer append in vgg16 model layer

# create a model object
model = Model(inputs=vgg.input, outputs=prediction) # give input is vgg16 put and outputs is prediction to our vgg16 model

# view the structure of the model
model.summary() # here we check summary of our model

# so we used the vgg16 model.

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
# compile

from keras.preprocessing.image import ImageDataGenerator

# -------this statements we use for image argumentation-------------


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Datasets/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Datasets/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''
#--------------------------------------------------------------------
#  here fit the model to our data
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# here plot loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# here plot accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

model.save('facefeatures_new_model.h5') # so here we are saving this model into our locsl folder...

