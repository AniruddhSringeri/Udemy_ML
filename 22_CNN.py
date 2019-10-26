#Convolutional Neural Network
#Here, we classify images into cats and dogs.

#Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential() 

#Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding second convolution layer to increase accuracy of model
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flattening
classifier.add(Flatten())

#Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling the CNN by implementing the stochastic gradient descent algorithm and the performance metric
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images

#Image Augmentation(It helps in enriching our limited dataset so that overfitting does not take place. Image Augmentation consists of making changes or augmentaions to separate batches of images while training, like rotation, shearing, flipping, shifting,...therefore, our model will be better trained.)
#This step can be done by taking code snippets from the keras documentation website.
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=1, validation_data=test_set, validation_steps=2000)