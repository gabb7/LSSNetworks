# ------------------------------------------------------------------------------
#
# Implementation of a convolutional autoencoder that can be trained on the
# Lumbar Spine Stenosis dataset for embedding the 3D images in a smaller space.
# The architecture is:
# - First convolutional layer
# - Max pooling
# - Second convolutional layer
# - Max pooling
# - First fully connected layer
# - Second fully connected layer (bottleneck)
# - Third fully connected layer (same size as first one)
# - Max unpooling
# - First deconvolutional layer
# - Max unpooling
# - Second deconvolutional layer
# The code obtained at the bottleneck stage with a trained autoencoder can be
# fed to binary classifiers, methods for encoding images and save the resulting
# vectors are provided.
#
# Copyright: 2017, Gabriele Abbati, University of Oxford
#
# ------------------------------------------------------------------------------


# Libraries
import tensorflow as tf
import numpy as np
from AutoEncoderImageDataset import AutoEncoderImageDataset
from skimage import io
import os.path


# ------------------------------------------------------------------------------



class ConvAutoEncoder(object):


	# Constructor. Arguments:
	# - path_to_train_directory: path-to-directory with which the DataSet
	#   attribute is initialized. It contains the training data;
 	# - batch_size: size of the training batch;
	# - image_size: size of the 3D images, which are image_size x image_size x
	#   n_channels;
	# - n_channels: 3rd dimension of the 3D images;
	# - learning_rate: initial learning rate (the network uses AdaGrad);
	# - beta: coefficient for regularization term in loss function;
	# - n_masks1: number of masks in the first convolutional layer and second
	#   deconvolutional layer;
	# - n_masks2: number of masks in the second convolutional layer and first
	#   deconvolutional layer;
	# - first_layer_size: number of nodes in the first and third fully
	#   connected layers;
	# - second_layer_size: number of nodes in the second fully connected layer,
	#   the bottleneck;
	# - dropout: probability to keep a connection in dropout stage.
	def __init__(self, path_to_train_directory, batch_size, image_size, \
			n_channels=1, learning_rate=1e-4, beta=0.0, n_masks1=64, \
			n_masks2=32, first_layer_size=1024, second_layer_size=128, \
			dropout=0.75):
		# Save parameters
		self.batch_size = batch_size
		self.image_size = image_size
		self.n_channels = n_channels
		self.DataSet = AutoEncoderImageDataset(path_to_train_directory,
				self.batch_size, self.image_size, self.n_channels)
		self.learning_rate = learning_rate
        self.beta = beta
		self.first_layer_size = first_layer_size
		self.second_layer_size = second_layer_size
		self.dropout = dropout
		self.n_input = self.image_size*self.image_size*self.n_channels
		self.n_masks1 = n_masks1
		self.n_masks2 = n_masks2
		# Some placeholder to be used later
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.keep_prob = tf.placeholder(tf.float32)
		# Weights and biases of the system
		self.encoder_weights = {
			# 5x5 conv, self.n_masks1 masks
			'wec1': tf.Variable(tf.random_normal(\
					[5, 5, self.n_channels, self.n_masks1],stddev=0.01) ),
			# 5x5 conv, self.n_masks2 masks
			'wec2': tf.Variable(tf.random_normal(\
					[5, 5, self.n_masks1, self.n_masks2],stddev=0.01) ),
			# fully connected: int(image_size/2)*int(image_size/2)*32 inputs,
			# self.first_layer_size outputs
			'wed1': tf.Variable(tf.random_normal([int(self.image_size/4)*\
					int(self.image_size/4)*self.n_masks2,\
					self.first_layer_size], stddev=0.01)),
			# fully connected: self.first_layer_size inputs,
			# self.second_layer_size outputs
			'wed2': tf.Variable(tf.random_normal([self.first_layer_size, \
					self.second_layer_size], stddev=0.01))
		}
		self.decoder_weights = {
			# fully connected: self.second_layer_size inputs,
			# self.first_layer_size outputs
			'wdd1': tf.Variable(tf.random_normal([self.second_layer_size, \
					self.first_layer_size],stddev=0.01)),
			# fully connected, int(image_size/2)*int(image_size/2)*n_masks
			# inputs
			'wdd2': tf.Variable(tf.random_normal([self.first_layer_size, \
					int(self.image_size/4) * int(self.image_size/4) * \
					self.n_masks2], stddev=0.01)),
			# 5x5 conv, self.n_masks2 masks
			'wdc2': tf.Variable(tf.random_normal([5, 5, self.n_masks1, \
					self.n_masks2], stddev=0.01)),
			# 5x5 conv, self.n_masks1 masks
			'wdc1': tf.Variable(tf.random_normal([5, 5, self.n_channels, \
					self.n_masks1], stddev=0.01))
		}
		# Biases for the encoder part of the network
		self.encoder_biases = {
			'bec1': tf.Variable(np.zeros(self.n_masks1,dtype='float32')),
			'bec2': tf.Variable(np.zeros(self.n_masks2,dtype='float32')),
			'bed1': tf.Variable(np.zeros(self.first_layer_size,\
					dtype='float32')),
			'bed2': tf.Variable(np.zeros(self.second_layer_size,\
					dtype='float32'))
		}
		# Biases for the decoder part of the network
		self.decoder_biases = {
			'bdd1': tf.Variable(np.zeros(self.first_layer_size,\
					dtype='float32')),
			'bdd2': tf.Variable(np.zeros(int(self.image_size/4)*\
					int(self.image_size/4)*self.n_masks2, \
					dtype='float32')),
			'bdc2': tf.Variable(np.zeros(self.n_masks1,dtype='float32')),
			'bdc1': tf.Variable(np.zeros(self.n_channels,dtype='float32'))
		}
		# Saver, needed to save the system after the training
		self.system_saver = tf.train.Saver()
		self.weights_saver = tf.train.Saver( \
				[ self.encoder_weights['wec1'],
				  self.encoder_weights['wec2'],
				  self.encoder_weights['wed1'],
				  self.encoder_weights['wed2'],
				  self.encoder_biases['bec1'] ,
				  self.encoder_biases['bec2'] ,
				  self.encoder_biases['bed1'] ,
				  self.encoder_biases['bed2']
				] )
		return


	# Dump weights values, to be later used to initialize the convolutional
	# neural network. Arguments:
	# - restore_directory: directory in which the weights are saved;
	# - session: current session in which the computation is performed, needed
	#   by TensorFlow.
	def dump_weights(self, restore_directory, session):
		weights_folder = restore_directory + "/" + "weights"
		if not os.path.exists(weights_folder):
			os.makedirs(weights_folder)
		self.weights_saver.save(session, weights_folder + "/" + "weights.ckpt")
		print "Weights saved"
		return


	# Directly read training and test sets from augmented data folder.
	# Arguments:
	# - path_to_training_directory: path to augmented data folder.
	def read_augmented_data(self, path_to_training_directory):
		self.DataSet.read_augmented_data(path_to_training_directory)
		return


	# Wrapper to 2D convolution (performs 2D convolution, adds the biases and
	# feed the results to the ReLU activation functions). Arguments:
	# - x: layer input;
	# - W: layer weights;
	# - b: layer biases;
	# - strides: stride of the sliding window for each dimension of the input x.
	def conv2d(self, x, W, b, strides=1):
		conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],\
				padding='SAME')
		conv = tf.nn.bias_add(conv, b)
		return tf.nn.relu(conv)


	# Wrapper to first 2D "deconvolution" (performs 2D deconvolution, adds the
	# biases and feed the results to the ReLU activation functions) Arguments:
	# - x: layer input;
	# - W: layer weights;
	# - b: layer biases;
	# - strides: stride of the sliding window for each dimension of the input x.
	def first_deconv2d(self, x, W, b, strides=1):
		deconv = tf.nn.conv2d_transpose(x, W, strides=[1, strides, strides, 1],\
				output_shape=[self.batch_size, self.image_size/2, \
				self.image_size/2, self.n_masks1], padding='SAME')
		deconv = tf.nn.bias_add(deconv, b)
		return tf.nn.relu(deconv)


	# Wrapper to second 2D "deconvolution" (performs 2D deconvolution, adds the
	# biases and feed the results to the ReLU activation functions) Arguments:
	# - x: layer input;
	# - W: layer weights;
	# - b: layer biases;
	# - strides: stride of the sliding window of the input x.
	def second_deconv2d(self, x, W, b, strides=1):
		deconv = tf.nn.conv2d_transpose(x, W, strides=[1, strides, strides, 1],\
				output_shape=[self.batch_size, self.image_size, \
				self.image_size, self.n_channels], padding='SAME')
		deconv = tf.nn.bias_add(deconv, b)
		return tf.nn.relu(deconv)


	# Wrapper to 2D Max-Pooling operation. Arguments:
	# - x: layer input;
	# - k: stride of the sliding window of the input x.
	def maxpool2d(self, x, k=2):
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
				padding='SAME')


	# Encode the image into a self.second_layer_size-long vector. Arguments:
	# - x: network input;
	# - weights: encoder weights;
	# - biases: encoder biases;
	# - dropout: probability to keep a connection at dropout stage.
	def encoder(self, x, weights, biases, dropout):
		# Reshape input picture
		x = tf.reshape(x, shape=[-1, self.image_size, self.image_size, \
				self.n_channels])
		# 1st convolution layer
		conv1 = self.conv2d(x, weights['wec1'], biases['bec1'])
		# Max Pooling (down-sampling)
		conv1 = self.maxpool2d(conv1, k=2)
		# Convolution Layer
		conv2 = self.conv2d(conv1, weights['wec2'], biases['bec2'])
		# Max Pooling (down-sampling)
		conv2 = self.maxpool2d(conv2, k=2)
		# First fully connected layer
		# Reshape conv1 output to fit fully connected layer input
		fc1 = tf.reshape(conv2, [-1, weights['wed1'].get_shape().as_list()[0]])
		fc1 = tf.add(tf.matmul(fc1, weights['wed1']), biases['bed1'])
		fc1 = tf.nn.relu(fc1)
		# Apply Dropout
		fc1 = tf.nn.dropout(fc1, dropout)
		# Second fully connected layer
		fc2 = tf.add(tf.matmul(fc1, weights['wed2']), biases['bed2'])
		fc2 = tf.nn.relu(fc2)
		return fc2


	# Decode the image from a self.second_layer_size-long vector into a
	# reconstruction of the image, which has the same structural characteristics
	# of the original one. Arguments:
	# - code: bottleneck output;
	# - weights: decoder weights;
	# - biases: decoder biases;
	# - dropout: probability to keep a connection at dropout stage.
	def decoder(self, code, weights, biases, dropout):
		# First fully connected layer
		fc1 = tf.add(tf.matmul(code, weights['wdd1']), biases['bdd1'])
		fc1 = tf.nn.relu(fc1)
		# Second fully connected layer
		fc2 = tf.add(tf.matmul(fc1, weights['wdd2']), biases['bdd2'])
		fc2 = tf.nn.relu(fc2)
		fc2 = tf.nn.dropout(fc2, dropout)
		# Reshaping
		fc2 = tf.reshape(fc2,[-1, int(self.image_size/4), \
				int(self.image_size/4), self.n_masks2])
		# First Unpooling (up-sampling, at the moment implemented as
		# an image resizing)
		unmax1 = tf.image.resize_images(fc2, self.image_size/2, self.image_size/2,\
				method=1)
		# First Deconvolution
		unconv1 = self.first_deconv2d(unmax1, weights['wdc2'], biases['bdc2'])
		# Second Unpooling (up-sampling, at the moment implemented as
		# an image resizing)
		unmax2 = tf.image.resize_images(unconv1,self.image_size,self.image_size,\
				method=1)
		# Second Deconvolution
		unconv2 = self.second_deconv2d(unmax2, weights['wdc1'], biases['bdc1'])
		return unconv2


	# Reconstruct the original image by learning the identity function.
	# Arguments:
	# - x: network input;
	# - encoder_weights: weights of the encoder half of the network;
	# - decoder_weights: weights of the decoder half of the network;
	# - encoder_biases: biases of the encoder half of the network;
	# - decoder_biases: biases of the decoder half of the network;
	# - dropout: probability to keep a connection at dropout stage.
	def reconstruct_image(self, x, encoder_weights, decoder_weights, \
			encoder_biases, decoder_biases, dropout):
		code = self.encoder(x, encoder_weights, encoder_biases, dropout)
		decode = self.decoder(code, decoder_weights, decoder_biases, dropout)
		return decode


	# Build the model (cost function, objective, optimizer)
	def build_model(self):
		# Reconstructed image
		self.reconstructed = self.reconstruct_image(self.x, \
				self.encoder_weights, self.decoder_weights, \
				self.encoder_biases, self.decoder_biases, self.dropout)
		# Quadratic pixelwise cost
		# if beta != 0.0, then insert the regularization term
        if self.beta == 0.0:
		    self.pixelwise_cost = tf.square(self.reconstructed - \
					tf.reshape(self.x, shape=[-1, self.image_size, \
					self.image_size, self.n_channels]))
		else:
            self.pixelwise_cost = tf.square(self.reconstructed - \
					tf.reshape(self.x, shape=[-1, self.image_size, \
					self.image_size, self.n_channels])) + \
                    self.beta*(tf.nn.l2_loss(self.encoder_weights['wec1']) + \
                    tf.nn.l2_loss(self.encoder_weights['wec2']) + \
                    tf.nn.l2_loss(self.encoder_weights['wed1']) + \
                    tf.nn.l2_loss(self.encoder_weights['wed2']) + \
                    tf.nn.l2_loss(self.decoder_weights['wdc1']) + \
                    tf.nn.l2_loss(self.decoder_weights['wdc2']) + \
                    tf.nn.l2_loss(self.decoder_weights['wdd1']) + \
                    tf.nn.l2_loss(self.decoder_weights['wdd2']))
		# Mean cost (display purposes)
		self.cost = tf.reduce_mean(tf.square(self.reconstructed - \
				tf.reshape(self.x, shape=[-1, self.image_size, \
				self.image_size, self.n_channels])))
		# Define the optimizer
		self.optimizer = tf.train.AdagradOptimizer(learning_rate = \
				self.learning_rate).minimize(self.pixelwise_cost)
		# Encoded test images
		self.encodeImages = self.encoder(self.x, self.encoder_weights, \
			self.encoder_biases, self.keep_prob)
		return


	# Initialize all variables (required by TensorFlow)
	def initialize_variables(self):
		self.init = tf.initialize_all_variables()
		return


	# Create summary writer (for post-training profiling). Arguments:
	# - restore_directory: directory in which the out files describing the
	#   training are contained;
	# - session: session in which the computations are performed, required by
	#   TensorFlow.
    def create_writer(self, restore_directory, session):
		# If not present, create the directory /logs
        if restore_directory != None:
            log_directory = restore_directory + '/logs'
        else:
            log_directory = '/logs'
		# Create training writer (tf summary writer)
        self.trainWriter = tf.train.SummaryWriter(log_directory,\
                session.graph)
        return


	# Create summaries for the summary writer. Arguments:
	# - restore_directory: directory in which the weights are saved.
    def create_scalar_summaries(self, restore_directory):
		# If not present, create the directory /logs
        if restore_directory != None:
            log_directory = restore_directory + '/logs'
        else:
            log_directory = '/logs'
		# Save cost function value with a saver
        with tf.name_scope('cost_function'):
            tf.scalar_summary(log_directory + '/cost_function/', self.cost)
        self.merged_summaries = tf.merge_all_summaries()
        return


	# Main function for training the convolutional autoencoder. Arguments:
	# - training_iters: number of data points that are fed to the network
	#   during training;
	# - display_step: number of step between printings of cost function on
	#   stdout;
	# - restore: binary variable, if True the weights contained in
	#   restore_directory are loaded and training is started from there. If
	#   False weights are initialized randomly;
	# - restore_directory: directory containing the potential saved weights to
	#   be restored it restore is True;
	# - save_step: number of steps between savings of network through Savers;
	# - test_step: number of steps between displaying of accuracies on the test
	#   set;
	# - save_weights: if True the weights of the encoder are saved for them to
	#   be used afterwards to initialize the weights of a convolutional neural
	#   network.
	# - summary_step: if different from None, number of steps between savings
	#   of the function values through Savers
	def train(self, training_iters=100000, display_step=5, restore=0, \
			restore_directory=None, save_step=1000, test_step=10, \
			save_weights=True, summary_step=None):
		# Build model and initialize variables
		self.build_model()
		self.initialize_variables()
		# Create directories for saving the network, if needed and not present,
		# and save the name of the checkpoint files, or load old ones
        if restore_directory != None:
            if not os.path.exists(restore_directory):
                    os.makedirs(restore_directory)
            if not os.path.exists(restore_directory + "/logs"):
                    os.makedirs(restore_directory + "/logs")
            restoreFile = restore_directory + "/AutoEncoder.ckpt"
        else:
            if not os.path.exists("/logs"):
                os.makedirs("/logs")
            restoreFile = "./AutoEncoder.ckpt"
		# Start of TensorFlow session
		with tf.Session() as sess:
			# If the user wants to save summaries about network variables,
			# create the relative writers
            if summary_step != None:
                self.create_writer(restore_directory, sess)
                self.create_scalar_summaries(restore_directory)
			# Initialize the TensorFlow session
			sess.run(self.init)
			# Restore the network from previous run if needed
			if restore == 1:
				ckpt = tf.train.get_checkpoint_state(restore_directory)
				if ckpt and ckpt.model_checkpoint_path:
					self.system_saver.restore(sess,ckpt.model_checkpoint_path)
				self.DataSet.read_training_test_set_list()
				self.DataSet.read_last_batch_image()
			step = 1
			# Keep training until reach max iterations
			while step * self.batch_size < training_iters:
				# Load the next batch
				batch_x = self.DataSet.next_batch()
                # Run optimization op (backprop)
				sess.run(self.optimizer, feed_dict={self.x: batch_x, \
						self.keep_prob: self.dropout})
				# Print loss function if needed
				if step % display_step == 0:
					# Calculate batch loss and accuracy
					loss = sess.run(self.cost, feed_dict={self.x: batch_x, \
							self.keep_prob: 1.})
					print "Iter " + str(step*self.batch_size) + \
							", Minibatch Loss = " + "{:.6f}".format(loss)
				# Save network state if needed
				if step % save_step == 0:
					self.system_saver.save(sess, restoreFile)
					print "System Saved"
				# Test network accuracy on test set
				if step % test_step == 0:
					batch_x = self.DataSet.next_test_batch()
					loss = sess.run(self.cost, feed_dict={self.x: batch_x, \
							self.keep_prob: 1. })
					print "Error on test set batch: ", loss
				# Save function values and weights if needed
                if summary_step != None:
                    if step % summary_step == 0:
					    summary = sess.run(self.merged_summaries,\
								feed_dict={self.x: batch_x, \
								self.keep_prob: 1.})
                        self.trainWriter.add_summary(summary, \
								step*self.batch_size)
				step += 1
			batch_x = self.DataSet.next_test_batch()
			self.originals = batch_x
			self.test_images = sess.run(self.reconstructed, \
					feed_dict = {self.x: batch_x, self.keep_prob: 1.} )
			# Save system before ending training
			self.system_saver.save(sess, restoreFile)
			# Save weights if needed
			if save_weights == True:
				self.dump_weights(restore_directory,sess)
		print "Training Completed"


    # Save codes and targets of the images that will be later used for the
    # training of a binary classifier. Arguments:
	# - imagesDirectory: path to directory containing the 3D images that have to
	#   be coded by the autoencoder;
	# - restore: binary variable, if True the autoencoder will be initialized
	#   with a state coming from a previous run;
	# - restore_directory: directory in which the checkpoint files needed for
	#   restoring the autoencoder are.
	def build_code_dataset(self, imagesDirectory, restore=0, \
			restore_directory=None):
		# Build model and initialize variables
		self.build_model()
		self.initialize_variables()
		# Check if restore directory actually exists
        if restore_directory != None:
            if not os.path.exists(restore_directory):
                os.makedirs(restore_directory)
            restoreFile = restore_directory + "/AutoEncoder.ckpt"
        else:
            restoreFile = "./AutoEncoder.ckpt"
		# Start TensorFlow session
		with tf.Session() as sess:
			# TensorFlow initialization
			sess.run(self.init)
			# Restore the autoencoder if needed
			if restore != None:
				self.system_saver.restore(sess, restoreFile)
				self.DataSet.read_training_test_set_list()
				self.DataSet.read_last_batch_image()
			# Do a couple batches training (this is needed because it is not
			# possible to save the right learning rate found through AdaGrad)
            for n in range(30):
				# Load next batch
		        batch_x = self.DataSet.next_batch()
                # Run optimization op (backprop)
                sess.run(self.optimizer, feed_dict={self.x: batch_x, \
						self.keep_prob: self.dropout})
				# Compute loss function
		    	loss = sess.run(self.cost, feed_dict={self.x: batch_x, \
						self.keep_prob: 1.})
            # Put the list of all data images in one place
			lstFilesNpy = []
			for dirName, subdirList, fileList in os.walk(imagesDirectory):
				for filename in fileList:
					if ".npy" in filename.lower():
						lstFilesNpy.append(os.path.join(dirName,filename))
			lstFilesNpy = sorted(lstFilesNpy)
			NImages = len(lstFilesNpy)
			NBatches = int(NImages / self.batch_size)
			# Start coding the images
			imageCodes = []
			for n in range(NBatches):
				# Create the Batch to feed the encoder
				batch_x = []
				for i in range(n*self.batch_size, (n+1)*self.batch_size):
					struct = np.load(lstFilesNpy[i])
					struct = struct.astype(np.float32)
					batch_x.append(struct)
                batch_x = np.asarray(batch_x).astype(np.float32)
				batch_x = np.reshape(batch_x, (self.batch_size, self.image_size*\
						self.image_size*self.n_channels))
				# Actually feed the batch to the autoencoder
				tmpCodes = sess.run(self.encodeImages, \
						feed_dict={self.x: batch_x, self.keep_prob: 1.} )
				# Append coded batch to the total image codes structure
                for j in range(self.batch_size):
                    imageCodes.append(tmpCodes[j,:])
            # Compute last batch (could be of a smaller size obviously)
			batch_x = []
			for i in range(NBatches*self.batch_size,NImages):
				struct = np.load(lstFilesNpy[i])
				struct = struct.astype(np.float32)
				batch_x.append(struct)
            batch_x = np.asarray(batch_x).astype(np.float32)
			batch_x = np.reshape(batch_x, (NImages - NBatches*self.batch_size, \
					self.image_size*self.image_size*self.n_channels))
			tmpCodes = sess.run(self.encodeImages, feed_dict={self.x: batch_x,
					self.keep_prob: 1.} )
			# Append the last batch to the total image codes structure
            for j in range(tmpCodes.shape[0]):
                imageCodes.append(tmpCodes[j,:])
			# Format the struct and save it as .npy file
	    	imageCodes = np.asarray(imageCodes).astype(np.float32)
	    	imageCodesFile = "imageCodes" + \
	            str(self.second_layer_size) + ".npy"
	        np.save(imageCodesFile, imageCodes)
		return
