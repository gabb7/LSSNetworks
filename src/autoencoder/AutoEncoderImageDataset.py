# ------------------------------------------------------------------------------
#
# Implementation of a dataset class for the convolutional autoencoder. It can
# load the available images from the indicated folder and provides data
# augmentation methods, as well as a next_batch method.
#
# Copyright: 2017, Gabriele Abbati, University of Oxford
#
# ------------------------------------------------------------------------------


# Libraries
import numpy as np
import tensorflow as tf
import csv
import os
import sys
from skimage import io
from random import uniform
from random import shuffle
from scipy.misc import imresize
from scipy.stats import bernoulli
from numpy.random import normal
from scipy import ndimage


# ------------------------------------------------------------------------------



class AutoEncoderImageDataset(object):


	# Constructor. Arguments:
	# - path_to_train_directory: path to the directory containing the 3D images
	#   used for training the network;
	# - batch_size: number of images a training batch is made of;
	# - image_size: size of the 3D images, which are image_size x image_size x
	#   n_channels;
	# - n_channels: 3rd dimension of the 3D images.
	def __init__(self, path_to_train_directory, batch_size, image_size, \
			n_channels):
		# Save variables for future use
		self.trainDirectory = path_to_train_directory
		self.batch_size = batch_size
		self.image_size = image_size
		self.n_channels = n_channels
		self.lastTestImage = 0
		return


	# Crops and convert to the right size (image_size x image_size) all the
	# 3D images in a struct series. Arguments:
	# - dataStruct: 3D image struct.
	def crop_and_resize(self, dataStruct):
		imageCenter = [int(dataStruct.shape[0]/2),int(dataStruct.shape[1]/2)]
		radius = [min(int(dataStruct.shape[0]/4), int(self.image_size/2)), \
				min(int(dataStruct.shape[1]/4), int(self.image_size/2)) ]
		imageCoords = [[imageCenter[0] - radius[0],imageCenter[0] + radius[0]],\
				[imageCenter[1] - radius[1],imageCenter[1] + radius[1]] ]
		self.n_channels = dataStruct.shape[2]
		new_image = np.zeros([self.image_size,self.image_size,self.n_channels])
		for i in range(self.n_channels):
			tmpImage = dataStruct[\
					imageCoords[0][0]:imageCoords[0][1],\
					imageCoords[1][0]:imageCoords[1][1],i]
			new_image[:,:,i] = imresize(tmpImage,\
					(self.image_size, self.image_size))
		return new_image


	# Normalizes image based on its maximum for each slide. Arguments:
	# - dataStruct: 3D image struct.
	def max_normalize(self, dataStruct):
		for i in range(self.n_channels):
			max_value = np.max(dataStruct[:,:,i])
			dataStruct[:,:,i] = dataStruct[:,:,i] / max_value
		return


	# Flips image on the Sagittal axis after having decided with a Bernoulli
	# Random variable. Arguments:
	# - dataStruct: 3D image struct.
	def random_sagittal_flip(self, dataStruct):
		if bernoulli.rvs(.5):
			for i in range(self.n_channels):
				dataStruct[:,:,i] = np.fliplr(dataStruct[:,:,i])
		return


	# Randomly rotates the pictures of an angle that is uniformly distributed
	# in the tange [-10,10] (degrees). Arguments:
	# - dataStruct: 3D image struct.
	def random_rotation(self, dataStruct):
		angle = uniform(-10.0, 10.0)
		for i in range(self.n_channels):
			dataStruct[:,:,i] = ndimage.rotate(dataStruct[:,:,i],angle,reshape=False)
		return


	# Adds Gaussian random noise to the picture (std dev = 0.05). Arguments:
	# - dataStruct: 3D image struct.
	def random_gaussian_noise(self, dataStruct):
		ImageSize = [ dataStruct.shape[0], dataStruct.shape[1] ]
		for i in range(ImageSize[0]):
			for j in range(ImageSize[1]):
				for n in range(self.n_channels):
					dataStruct[i,j,n] = min(1.0, max([dataStruct[i,j,n]+\
							normal(0.0,0.05), 0.0 ]))
		return


	# Randomly perturbate brightness and contrast to make the network
	# insensitive to changes of this kind. Arguments:
	# - dataStruct: 3D image struct.
	def random_brightness(self, dataStruct):
		return tf.image.random_brightness(dataStruct, 0.05)


	# Randomly invert pictures order. Arguments:
	# - dataStruct: 3D image struct.
	def random_invert_order(self, dataStruct):
		if bernoulli.rvs(.5):
			for i in range(int(self.n_channels/2)):
				tmp = dataStruct[:,:,i]
				dataStruct[:,:,i] = dataStruct[:,:,self.n_channels-1-i]
				dataStruct[:,:,self.n_channels-1-i] = tmp
		return


	# Perform data augmentation on the training data using the previous
	# functions.Arguments:
	# - path_to_augmented_train_dir: path to directory that will contain the new
	#   augmented dataset;
	# - DataAugFactor: data augmentation factor, number of times a single 3D
	def augment_training_data(self, path_to_augmented_train_dir, DataAugFactor):
		# Save variables
		self.augmTrainDirectory = path_to_augmented_train_dir
		self.AugFactor = DataAugFactor
		# Get all file names
		npyFileList = []
		for dirName, subdirList, fileList in os.walk(self.trainDirectory):
			for filename in fileList:
				if ".npy" in filename.lower():
					npyFileList.append(os.path.join(dirName,filename))
		# Creation of data folder which will contain data structures
		if not os.path.exists(self.augmTrainDirectory):
			os.makedirs(self.augmTrainDirectory)
		# Data augmentation: Open .npy file, create self.AugFactor different
		# structs for each one, save them into the right folder
		# (self.augmTrainDirectory)
		index_level = 0
		for filename in npyFileList:
			levelImage = np.load(filename)
			levelImage = levelImage.astype(np.float32)
			for n in range(self.AugFactor):
				struct = np.copy(levelImage)
				self.random_rotation(struct)
				struct = self.crop_and_resize(struct)
				self.random_sagittal_flip(struct)
				self.random_invert_order(struct)
				self.max_normalize(struct)
				self.random_gaussian_noise(struct)
				self.random_brightness(struct)
				npyFilePath = self.augmTrainDirectory + "/" + \
					"levelData_" + str(index_level)
				np.save(npyFilePath, struct)
				index_level = index_level + 1
		# Get all new file names
		self.read_augmented_data(self.augmTrainDirectory)
		return


	# Directly read training and test sets from augmented data folder (after
	# data augmentation). Arguments:
	# - path_to_augmented_train_dir: path to the directory containing the
	#   augmented data structs.
	def read_augmented_data(self, path_to_augmented_train_dir):
		self.augmTrainDirectory = path_to_augmented_train_dir
		# Get all new file names
		npyFileList = []
		for dirName, subdirList, fileList in os.walk(self.augmTrainDirectory):
			for filename in fileList:
				if ".npy" in filename.lower():
					npyFileList.append(os.path.join(dirName,filename))
		# Create Training and test set, using 90% - 10% ratio
		self.NTotLevels = len(npyFileList)
		self.NtrainingData = int(0.9 * self.NTotLevels)
		self.NtestData = self.NTotLevels - self.NtrainingData
		shuffle(npyFileList)
		shuffle(npyFileList)
		self.trainingSet = npyFileList[0:self.NtrainingData]
		self.testSet = npyFileList[self.NtrainingData:]
		self.lastBatchImage = 0
		self.dump_training_test_set_list()
		self.dump_last_batch_image()
		return


	# Dump information about training and test set on .csv files, in order to
	# know where to start training again
	def dump_training_test_set_list(self):
		# Training set
		with open('training_set.csv', 'wb') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',',\
					quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for filename in self.trainingSet:
				filewriter.writerow([filename])
		# Test set
		with open('test_set.csv', 'wb') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',',\
					quotechar='|', quoting=csv.QUOTE_MINIMAL)
			for filename in self.testSet:
				filewriter.writerow([filename])
			return


	# Dump information about the first image of the last batch used in training
	def dump_last_batch_image(self):
		# Last item used
		with open('last_batch.csv', 'wb') as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',',\
					quotechar='|', quoting=csv.QUOTE_MINIMAL)
			filewriter.writerow([self.lastBatchImage])
		return


	# Load training and test set list dumped with the previous method
	def read_training_test_set_list(self):
		# Training set
		self.trainingSet = []
		with open('training_set.csv', 'rb') as f:
			filereader = csv.reader(f)
			for row in filereader:
				self.trainingSet.append(row[0])
		self.NtrainingData = len(self.trainingSet)
		# Test set
		self.testSet = []
		with open('test_set.csv', 'rb') as f:
			filereader = csv.reader(f)
			for row in filereader:
				self.testSet.append(row[0])
		self.NtestData = len(self.testSet)
		return


	# Dump information about the first image of the last batch used in training
	def read_last_batch_image(self):
		# Last item used
		with open('last_batch.csv', 'rb') as f:
			filereader = csv.reader(f)
			for row in filereader:
				self.lastBatchImage = int(row[0])
		return


	# Returns the next batch used for training the network in the format
	# requested by TensorFlow (numpy float32)
	def next_batch(self):
		images = []
		if self.lastBatchImage + self.batch_size <= self.NtrainingData:
			for i in range(self.lastBatchImage, self.lastBatchImage + \
					self.batch_size):
				levelImage = np.load(self.trainingSet[i])
				levelImage = levelImage.astype(np.float32)
				images.append(levelImage)
			self.lastBatchImage = self.lastBatchImage + self.batch_size
		else:
			for i in range(self.lastBatchImage, self.NtrainingData):
				levelImage = np.load(self.trainingSet[i])
				levelImage = levelImage.astype(np.float32)
				images.append(levelImage)
			self.lastBatchImage = 0
			missingNImages = self.batch_size - len(images)
			for i in range(self.lastBatchImage, missingNImages):
				levelImage = np.load(self.trainingSet[i])
				levelImage = levelImage.astype(np.float32)
				images.append(levelImage)
			self.lastBatchImage = missingNImages
		images = np.asarray(images).astype(np.float32)
		batch_x = np.reshape(images, (-1, self.image_size*\
				self.image_size*self.n_channels) )
		return batch_x


	# Returns the next batch used for training the network in the format
	# requested by TensorFlow (numpy float32)
	def next_test_batch(self):
		images = []
		if self.lastTestImage + self.batch_size <= self.NtestData:
			for i in range(self.lastTestImage, self.lastTestImage + \
					self.batch_size):
				levelImage = np.load(self.testSet[i])
				levelImage = levelImage.astype(np.float32)
				images.append(levelImage)
			self.lastTestImage = self.lastTestImage + self.batch_size
		else:
			for i in range(self.lastTestImage, self.NtestData):
				levelImage = np.load(self.testSet[i])
				levelImage = levelImage.astype(np.float32)
				images.append(levelImage)
			self.lastTestImage = 0
			missingNImages = self.batch_size - len(images)
			for i in range(self.lastTestImage, missingNImages):
				levelImage = np.load(self.trainingSet[i])
				levelImage = levelImage.astype(np.float32)
				images.append(levelImage)
			self.lastTestImage = missingNImages
		images = np.asarray(images).astype(np.float32)
		batch_x = np.reshape(images, (-1, self.image_size*\
				self.image_size*self.n_channels) )
		return batch_x
