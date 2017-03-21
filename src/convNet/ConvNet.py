# ------------------------------------------------------------------------------
#
# Implementation of a convolutional neural networks that can be trained on the
# Lumbar Spine Stenosis dataset for surgical planning. The architecture is:
# - First convolutional layer
# - Max pooling
# - Second convolutional layer
# - Max pooling
# - First fully connected layer
# - Second fully connected layer
#
# Copyright: 2017, Gabriele Abbati, University of Oxford
#
# ------------------------------------------------------------------------------


# Libraries
import tensorflow as tf
import numpy as np
from ConvNetImageDataset import ConvNetImageDataset
import os.path
import sys
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------------------



class ConvNet(object):


	# Constructor. Arguments:
	# - path_to_train_directory: path-to-directory with which the DataSet
	#   attribute is initialized. It contains the training data;
 	# - batch_size: size of the training batch;
	# - image_size: size of the 3D images, which are image_size x image_size x
	#   n_channels;
	# - n_channels: 3rd dimension of the 3D images
	# - learning_rate: initial learning rate (the network uses AdaGrad)
	# - n_masks1: number of masks in the first convolutional layer
	# - n_masks2: number of masks in the second convolutional layer
	# - first_layer_size: number of nodes in the first fully connected layer
	# - second_layer_size: number of nodes in the second fully connected layer
	# - dropout: probability to keep a connection in dropout stage
	def __init__(self, path_to_train_directory, batch_size, image_size, \
			n_channels=1, learning_rate=1e-4, n_masks1=64, n_masks2=32, \
			first_layer_size=1024, second_layer_size=128, n_classes=2,\
			dropout=0.75):
		# Save parameters
		self.batch_size = batch_size
		self.image_size = image_size
		self.n_channels = n_channels
		self.DataSet = ConvNetImageDataset(path_to_train_directory,
				self.batch_size, self.image_size, self.n_channels)
		self.learning_rate = learning_rate
		self.first_layer_size = first_layer_size
		self.second_layer_size = second_layer_size
		self.dropout = dropout
		self.n_input = self.image_size*self.image_size*self.n_channels
		self.n_masks1 = n_masks1
		self.n_masks2 = n_masks2
		self.n_classes = n_classes
		# Some placeholder to be used later
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.y = tf.placeholder(tf.float32, [None, self.n_classes])
		self.keep_prob = tf.placeholder(tf.float32)
		# Weights of the network
		self.weights = {
			# 5x5 conv, 4 input, self.n_masks1 outputs
			'wc1': tf.Variable(tf.random_normal(\
					[5, 5, self.n_channels, self.n_masks1],stddev=0.01) ),
			# 5x5 conv, 4 input, self.n_masks2 outputs
			'wc2': tf.Variable(tf.random_normal(\
					[5, 5, self.n_masks1, self.n_masks2],stddev=0.01) ),
			# fully connected, int(image_size/2)*int(image_size/2)*32 inputs,
			# 1024 outputs
			'wfc1': tf.Variable(tf.random_normal([int(self.image_size/4)*\
					int(self.image_size/4)*self.n_masks2,\
					self.first_layer_size], stddev=0.01)),
			# fully connected, 1024 inputs, 128 outputs
			'wfc2': tf.Variable(tf.random_normal([self.first_layer_size, \
					self.second_layer_size], stddev=0.01)),
			# out weights
			'out': tf.Variable( tf.random_normal( \
				[self.second_layer_size, n_classes] ) )
		}
		# Biases of the network
		self.biases = {
			'bc1': tf.Variable(np.zeros(self.n_masks1,dtype='float32')),
			'bc2': tf.Variable(np.zeros(self.n_masks2,dtype='float32')),
			'bfc1': tf.Variable(np.zeros(self.first_layer_size,\
					dtype='float32')),
			'bfc2': tf.Variable(np.zeros(self.second_layer_size,\
					dtype='float32')),
			'out': tf.Variable(np.zeros(n_classes,dtype='float32'))
		}
		# Savers, needed to save the system after training and restore
		# variable from autoencoder training
		self.system_saver = tf.train.Saver()
		self.weights_saver = tf.train.Saver( \
				[ self.weights['wc1'],
				  self.weights['wc2'],
				  self.weights['wfc1'],
				  self.weights['wfc2'],
				  self.biases['bc1'],
				  self.biases['bc2'],
				  self.biases['bfc1'],
				  self.biases['bfc2']
				] )
		return


	# Initialize the weights and biases with the values obtained from a previous
	# autoencoder training. Arguments:
	# - restore_directory: directory in which the weights are saved;
	# - session: current session in which the computation is performed, needed
	#   by TensorFlow.
	def load_weights(self,restore_directory, session):
		if restore_directory == None:
			weights_folder = "weights"
		else:
			weights_folder = restore_directory + "/" + "weights"
		self.weights_saver.restore(session, weights_folder + "/" + \
				"weights.ckpt")
		print "Weights restored"
		return


	# Read the pre-processed training and test sets from the augmented data
	# folder. Arguments:
	# - path_to_training_directory: path to augmented data folder.
	def read_augmented_data(self, path_to_training_directory):
		self.DataSet.read_augmented_data(path_to_training_directory)
		return


	# Wrapper to 2D convolution (performs 2D convolution, adds the biases and
	# feed the results to the ReLU activation functions). Arguments:
	# - x: layer input;
	# - W: layer weights;
	# - b: layer biases;
	# - strides: stride of the sliding window of the input x.
	def conv2d(self, x, W, b, strides=1):
		conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],\
				padding='SAME')
		conv = tf.nn.bias_add(conv, b)
		return tf.nn.relu(conv)


	# Wrapper to 2D Max-Pooling operation. Arguments:
	# - x: layer input;
	# - k: stride of the sliding window of the input x.
	def maxpool2d(self, x, k=2):
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
						  padding='SAME')


	# Perform a network training step with the input batch x.
	# - x: network input;
	# - weights: network weights;
	# - biases: network biases;
	# - dropout: probability to keep a connection at dropout stage.
	def inference(self, x, weights, biases, dropout):
		# Reshape input picture for computation purposes
		x = tf.reshape(x, shape=[-1, self.image_size, self.image_size, \
				self.n_channels])
		# 1st Convolution Layer
		conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
		# 1st Max Pooling (down-sampling)
		conv1 = self.maxpool2d(conv1, k=2)
		# 2nd Convolution Layer
		conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
		# 2nd Max Pooling (down-sampling)
		conv2 = self.maxpool2d(conv2, k=2)
		# Reshape conv2 output to fit fully connected layer input
		fc1 = tf.reshape(conv2, [-1, weights['wfc1'].get_shape().as_list()[0]])
		# 1st fully connected layer
		fc1 = tf.add(tf.matmul(fc1, weights['wfc1']), biases['bfc1'])
		fc1 = tf.nn.relu(fc1)
		# Apply Dropout
		fc1 = tf.nn.dropout(fc1, dropout)
		# 2nd fully connected layer
		fc2 = tf.add(tf.matmul(fc1, weights['wfc2']), biases['bfc2'])
		fc2 = tf.nn.relu(fc2)
		# Out layer
		out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
		return out


	# Build the model (cost function, objective, optimizer)
	def build_model(self):
		# Predicted Labels
		self.prediction = self.inference(self.x, \
				self.weights, self.biases, self.dropout)
		self.predicted_classes = tf.argmax(self.prediction, 1)
		# Cost function
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
				self.prediction, self.y))
		# Define the optimizer (AdaGrad)
		self.optimizer = tf.train.AdagradOptimizer(learning_rate = \
				self.learning_rate).minimize(self.cost)
		# Evaluate the model
		self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), \
				tf.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
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


	# Create weights writer, used to save and restore the weights of the
	# network. Arguments:
	# - restore_directory: directory in which the weights are saved;
	# - session: session in which the computations are performed, required by
	#   TensorFlow.
	def create_weights_writer(self, restore_directory, session):
		# If not present, create the directory /logs
		if restore_directory != None:
			log_directory = restore_directory + '/logs'
		else:
			log_directory = '/logs'
		# Create weights writer (tf summary writer)
		self.weightsWriter = tf.train.SummaryWriter(log_directory,\
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
		# Save cost function and accuracy values
		with tf.name_scope('cost_function'):
			tf.scalar_summary(log_directory + '/cost_function/', self.cost)
		with tf.name_scope('accuracy'):
			tf.scalar_summary(log_directory + '/accuracy/', self.accuracy)
		self.merged_summaries = tf.merge_all_summaries()
		return


	# Create summaries for the summary writer. This one saves some values
	# relative to the weights, in an attempt to visualize how the network is
	# after training. Arguments:
	# - restore_directory: directory in which the weights are saved.
	def create_weights_summaries(self, restore_directory):
		# If not present, create the directory /logs
		if restore_directory != None:
			log_directory = restore_directory + '/logs'
		else:
			log_directory = '/logs'
		# Save mean value of masks in 1st convolutional layer
		with tf.name_scope('conv1_mask_average'):
			conv1_mean = tf.reduce_mean(self.weights['wc1'] ,\
					reduction_indices=[0,1])
			conv1_mean = tf.reshape(conv1_mean, [1, \
					conv1_mean.get_shape().as_list()[0], \
					conv1_mean.get_shape().as_list()[1], 1])
			tf.image_summary(log_directory + '/conv1_mask_average/', \
					conv1_mean)
		# Save mean value of masks in 2nd convolutional layer
		with tf.name_scope('conv2_mask_average'):
			conv2_mean = tf.reduce_mean(self.weights['wc2'] ,\
					reduction_indices=[0,1])
			conv2_mean = tf.reshape(conv2_mean, [1, \
					conv2_mean.get_shape().as_list()[0], \
					conv2_mean.get_shape().as_list()[1], 1])
			tf.image_summary(log_directory + '/conv2_mask_average/', \
					conv2_mean)
		# Save 1st fully connected layer values
		with tf.name_scope('fc1'):
			fc1_weights = tf.reshape(self.weights['wfc1'], [1, \
					self.weights['wfc1'].get_shape().as_list()[0], \
					self.weights['wfc1'].get_shape().as_list()[1], 1])
			tf.image_summary(log_directory + '/fc1_weights/', \
					fc1_weights)
		# Save 2nd fully connected layer values
		with tf.name_scope('fc2'):
			fc2_weights = tf.reshape(self.weights['wfc2'], [1, \
					self.weights['wfc2'].get_shape().as_list()[0], \
					self.weights['wfc2'].get_shape().as_list()[1], 1])
			tf.image_summary(log_directory + '/fc2_weights/', \
					fc2_weights)
		self.merged_summaries = tf.merge_all_summaries()
		return


	# Main function for training the convolutional neural network. Arguments:
	# - training_iters: number of data points that are fed to the network
	#   during training;
	# - display_step: number of step between printings of cost function and
	#   accuracy on stdout;
	# - restore: binary variable, if True the weights contained in
	#   restore_directory are loaded and training is started from there. If
	#   False weights are initialized randomly;
	# - restore_directory: directory containing the potential saved weights to
	#   be restored it restore is True;
	# - save_step: number of steps between savings of network through Savers;
	# - test_step: number of steps between displaying of accuracies on the test
	#   set;
	# - restore_weights: if True weights are restored from a previous run of the
	#   autoencoder. Different from restore in the sense that here only weights
	#   and biases are touched;
	# - summary_step: if different from None, number of steps between savings
	#   of the function values through Savers
	def train(self, training_iters=100000, display_step=5, restore=False, \
			restore_directory=None, save_step=1000, test_step=10,\
			restore_weights=False, summary_step=None):
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
			restoreFile = restore_directory + "/ConvNet.ckpt"
		else:
			if not os.path.exists("/logs"):
				os.makedirs("/logs")
			restoreFile = "./ConvNet.ckpt"
		# Start of TensorFlow session
		with tf.Session() as sess:
			# If the user wants to save summaries about network variables,
			# create the relative writers
			if summary_step != None:
				self.create_writer(restore_directory, sess)
				self.create_weights_writer(restore_directory, sess)
				self.create_scalar_summaries(restore_directory)
				self.create_weights_summaries(restore_directory)
			# Initialize the TensorFlow session
			sess.run(self.init)
			# Restore the network from previous run if needed
			if restore == True:
				self.system_saver.restore(sess, restoreFile)
				self.DataSet.read_training_test_set_list()
				self.DataSet.read_last_batch_image()
			# Restore the weights from previous run if needed
			if restore_weights == True:
				self.load_weights(restore_directory,sess)
			step = 1
			# Keep training until reach max iterations
			while step * self.batch_size < training_iters:
				# Load the next batch, data and targets
				batch_x, batch_y = self.DataSet.next_batch()
				# Run optimization op (backprop)
				sess.run(self.optimizer, feed_dict={self.x: batch_x, \
						self.y: batch_y, \
						self.keep_prob: self.dropout})
				# Print loss function and accuracy if needed
				if step % display_step == 0:
					# Calculate batch loss and accuracy
					loss, acc  = sess.run([self.cost, self.accuracy],\
							feed_dict={self.x: batch_x, \
							self.y: batch_y, \
							self.keep_prob: 1.})
					print "Iter " + str(step*self.batch_size) + \
							", Minibatch Loss = " + \
							"{:.6f}".format(loss) + ", Training Accuracy = "+\
							"{:.5f}".format(acc)
				# Save network state if needed
				if step % save_step == 0:
					self.system_saver.save(sess, restoreFile)
					print "System Saved"
				# Test network accuracy on test set
				if step % test_step == 0:
					batch_x, batch_y = self.DataSet.next_test_batch()
					print "Testing Accuracy:", \
							sess.run(self.accuracy, feed_dict={self.x: batch_x, \
							self.y: batch_y, \
							self.keep_prob: 1. })
				# Save function values and weights if needed
				if summary_step != None:
					if step % summary_step == 0:
						summary = sess.run(self.merged_summaries,\
							feed_dict={self.x: batch_x, \
							self.y: batch_y, \
							self.keep_prob: 1.})
						self.trainWriter.add_summary(summary,step*self.batch_size)
						self.weightsWriter.add_summary(summary,step*self.batch_size)
				step += 1
			# Save system before ending training
			self.system_saver.save(sess, restoreFile)
		print "Training Completed"
		return


	# Test the network on the whole test set, returns the area under the ROC
	# curve. Arguments:
	# - restore_directory: directory containing the network state after a
	#   previous training
	def compute_test_error(self,restore_directory):
		# Build model and initialize variables
		self.build_model()
		self.initialize_variables()
		# Check if restore directory actually exists
		if not os.path.exists(restore_directory):
			sys.out("Restore directory not valid")
		restoreFile = restore_directory + "/ConvNet.ckpt"
		# Start TensorFlow session
		with tf.Session() as sess:
			# TensorFlow initialization
			sess.run(self.init)
			# Restore the network
			self.system_saver.restore(sess, restoreFile)
			# Read lists describing test and training set
			self.DataSet.read_training_test_set_list()
			self.DataSet.read_last_batch_image()
			# Compute Roc score for the whole test set
			RealTargets = []
			PredictedTargets = []
			step = 1
			NTestData = self.DataSet.NtestData
			while step * self.batch_size < NTestData:
				# Load new test batch
				batch_x, batch_y = self.DataSet.next_test_batch()
				# Predict labels
				Y_pred = sess.run(self.predicted_classes, feed_dict={self.x: batch_x,\
										self.keep_prob: 1.0})
				PredictedTargets = PredictedTargets + list(Y_pred)
				# Format true targets
				for target in batch_y:
					if target[0] == 1.0:
						RealTargets.append(0)
					elif target[1] == 1.0:
						RealTargets.append(1)
				step = step + 1
		# Compute scores: area under the ROC and total accuracy
		ROCscore = roc_auc_score(RealTargets, PredictedTargets)
		accuracy = float(np.sum(map(int, np.equal(RealTargets, \
				PredictedTargets)))) / float(len(RealTargets))
		# Print results on std out
		print "Scores obtained on the whole test dataset"
		print "Accuracy = ", accuracy
		print "Area under the ROC = ", ROCscore
	    return
