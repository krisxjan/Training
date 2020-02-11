from __future__ import print_function
import os
import numpy as np
import cv2
from cv2 import imread


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import os

DEFAULT_WIDTH = 32
DEFAULT_HEIGHT = 32

'''
Function Name: load_data()
Function Description: this function loads data from a designated dataset directory and returns
	the training set and testing set as numpy array, as well as the number of classes in the dataset
Parameters:
	- path: path of the dataset directory (default = './dataset')
	- max_x: width of the image (default = 32)
	- max_y: height of the image (default = 32)
	- prop: the proportion data reserved for testing set (default = 0.2)
Return Values:
	- x_train: trainind dataset
	- x_test: testing dataset
	- y_train: labels for the training set
	- y_test: labels for the testing set
	- num_classes: number of classes in the dataset
'''
def load_data(path = './dataset/', max_x = DEFAULT_WIDTH, max_y = DEFAULT_HEIGHT, prop = 0.2):
	x_train = np.empty([0, max_x, max_y, 3])
	x_test = np.empty([0, max_x, max_y, 3])

	y_train = np.empty([0])
	y_test = np.empty([0])
	num_classes = -1

	for dirpath, dirname, filename in os.walk(path):
		x_data=[]
		y_data=[]
		for f in filename:
			fp = os.path.join(dirpath, f)	# image file
			#print("file: ", fp)
			#print(dirpath)
			fp = os.path.join(dirpath, f)   # image file
			image = imread(fp)
                        #print("loading file: ", fp)
			image = cv2.resize(image, (max_y,max_x))

			if len(image.shape) == 3:
                            # image is rgb
				x_data.append(image)
				y_data.append(num_classes)

		if num_classes != -1:
			x_data = np.array(x_data)
			y_data = np.array(y_data)
			num_of_image = x_data.shape[0]

			num_of_test = int(num_of_image * prop)
			num_of_train = num_of_image - num_of_test

			x_data_train = x_data[0:num_of_train, :]
			x_data_test = x_data[num_of_train:, :]

			y_data_train = y_data[0:num_of_train]
			y_data_test = y_data[num_of_train:]

			x_train = np.concatenate((x_train, x_data_train), axis = 0)
			x_test = np.concatenate((x_test, x_data_test), axis = 0)

			y_train = np.concatenate((y_train, y_data_train), axis = 0)
			y_test = np.concatenate((y_test, y_data_test), axis = 0)
                                                                                                      
		num_classes += 1




	return (x_train, y_train), (x_test, y_test), num_classes



''' 
Function Name: load_model()
Function Description: this function builds the model 
Parameters:
	- num_classes: number of objects being trained 
Return Value:
	- model: object contraining the model, with weights loaded
'''
def load_model(num_classes):
	model = Sequential()

	# TODO: add a 2D convolution layer with 32 filters, and 6x6 kernal, make this the input layer
	model.add(Conv2D(32,(6,6),padding='same',input_shape=x_train.shape[1:]))
	# TODO: add a relu activation layer
	model.add(Activation('relu'))
	# TODO: add a batch normalization layer
	model.add(BatchNormalization())
	# TODO: add a 2D max pooling layer with 2x2 kernal
	model.add(MaxPooling2D(pool_size=(2,2)))
	# TODO: add a flatten layer
	model.add(Flatten())
	# TODO: add a fully-connected layer with 32 units and relu activation function
	model.add(Dense(32,activation='relu'))
	# TODO: add a dropout layer with 30% drop rate
	model.add(Dropout(0.3))
	model.add(Dense(num_classes, activation = 'softmax'))
	model.summary()

	return model


'''
Function Name: train_model()
Function Description: this function trains the model with hyper-parameters specified by as inputs to the
	function call.
Parameters:
	- model: neural network model created by load_model() function call
	- xTrain: feature vectors for training
	- yTrain: label vectors for training
	- xTest: feature vectors for validation 
	- yTest: label vectors for validation
	- num_classes: num of classes in the dataset (Integer)
	- batchSize: batch size to user per epoch (Integer)
	- max_epoches: number of forward and backword pass through the network
	- learningRage: learning rate used during gradient descent
	- outFile: name of the model to save the weights after training
Return Value:
	- model: trained model
'''
def train_model(model, xTrain, yTrain, xTest, yTest,
		num_classes, batchSize = 128, max_epoches = 250,learningRate = 0.001, outFile = 'personal_train.h5'):
	
	batch_size = batchSize
	maxepoches = max_epoches
	learning_rate = learningRate

	(x_train, y_train), (x_test, y_test) = (xTrain, yTrain),(xTest, yTest)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	
	#TODO: compile the model with 'categorical_crossentropy' as loss function and
	#			stocastic gradient descent optomizer with learning rate specified by 
	#			the input parameter and 'accuracy' metrics
	opt=keras.optimizers.SGD(lr=learning_rate)
	model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

	# TODO: train the model with (x_test, y_test) as validation data, with other hyper-parameters defined
	#			by the inputs to this function call

#	datagen.fit(x_train)
#	model.fit_generator(datagen.flow(x_train, y_train,
#                                    batch_size=batch_size),
#                        epochs=maxepochs,
#                        validation_data=(x_test, y_test))
	model.fit(x_train, y_train,batch_size=batch_size,epochs=maxepoches,
		validation_data=(x_test, y_test),shuffle=True)

	# TODO: save model weight to the file specified by the 'outFile' parameter
	model.save_weights(outfile)
	return model


if __name__ == '__main__':

	dataset_path = './dataset/' 

	num_classes = 0
	x_train = []
	x_test = []
	y_train = []
	y_test = []
	(x_train, y_train), (x_test, y_test), num_classes = load_data(path = dataset_path)

	# TODO: remove exit(-1) when load_data() is completed
	#exit(-1)


	# TODO: remove exit(-1) once load_model() is completed
	model = load_model(num_classes)
	# exit(-1)

	# TODO: remove exit(-1) once train_model() is completed
	model = train_model(model, x_train, y_train, x_test, y_test, num_classes)
	# exit(-1)

	predicted_x = model.predict(x_test)
	residuals = np.argmax(predicted_x,1)==y_test

	loss = sum(residuals)/len(residuals)
	print("The validation 0/1 loss is: ",loss)  






