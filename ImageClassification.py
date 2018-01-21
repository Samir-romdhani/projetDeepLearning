from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
import keras.optimizers as ko

import tensorflow as tf
import keras.backend as k

data_dir = 'data'
N_train = 1000
N_val = 400

img_width = 150
img_height = 150

model = Sequential()
# Importation Conv2D de keras.layers, 
# ceci pour effectuer l'opération de convolution, 
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), data_format="channels_last"))
model.add(Activation('relu'))
#importation MaxPooling2D de keras.layers, qui est utilisé pour l'opération de pooling
# Nous utilisons une fonction de Maxpool
model.add(MaxPooling2D(pool_size=(2, 2)))

# Importation Flatten de keras.layers, qui est utilisé pour 
#  la conversion de tous les tableaux bidimensionnels résultants en un seul vecteur.
model.add(Flatten())
# Importation Dense de keras.layers, qui est utilisé pour effectuer "the full connection of the neural network"
model.add(Dense(64,name='first', input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1,name='second'))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

import keras.preprocessing.image as kpi
train_datagen = kpi.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
valid_datagen = kpi.ImageDataGenerator(rescale=1./255)

#Nous divisons l'ensemble d'entraînement en batchs, 
#chaque epochs passe par tout l'ensemble d'entraînement.
#Chaque itération passe par batch.
batch_size = 100 # 1000/100
epochs = 50

#Un générateur qui va lire les images trouvées dans 'data / train'
train_generator = train_datagen.flow_from_directory(
        data_dir+"/train/",  #le répertoire
        target_size=(img_width, img_height), #toutes les images seront redimensionnées à 150x150
        batch_size=batch_size,
        class_mode='binary',
        classes=['cats','dogs'])
# Un générateur similaire, pour les données de validation
validation_generator = valid_datagen.flow_from_directory(
        data_dir+"/validation/",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
model.fit_generator(train_generator,
                    steps_per_epoch=N_train// batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=N_val// batch_size)
#
model.save('data\\model\\models_convolutional_network_%d_epochs_%d_batch_size.h5' %(epochs, batch_size))
#
score_conv_train = model.evaluate_generator(train_generator, N_train// batch_size)
score_conv_val = model.evaluate_generator(validation_generator, N_val //batch_size)
print('Train accuracy:', score_conv_train[1])
print('Test accuracy:', score_conv_val[1])