from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import random
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, regularizers, Lambda, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from keras import backend as K
totalStart = time.time()

'''
If you run this whole script it will build one model and save, but not display, 10 images, ~80 seconds on my laptop
Some of the code verification and dataset verification functions output to the console
'''

# ################## for saving plots, as well as code and dataset verification images
output_folder_path = ""
###################

"""
The two sets of classes from the assignment instructions are referred as img_set_A (training and testing) and img_set_B (testing only)
• img_set_A = ["top", "trouser", "pullover", "coat", "sandal", "ankle boot"]
• img_set_B = ["dress", "sneaker", "bag", "shirt"]
• positive pairs are labelled as 1, negative as 0. 
• 'class-integer' is used to refer the integer representing the class of a particular image

Variables:
:param output_folder_path: location to save plots and models
:param batch_sizes: list of integers, how many pairs to include in each training batch
:param epochs: number of epochs to train each model
:param pics: numpy array of 70000 greyscale images, each a 28*28 array, values 0 to 255, normalised after importation
:param categories: numpy array of 70000 integers, 0 to 9, corresponding to class of image at same index in pics
:param img_rows, img_cols: 28, dimensions of images
:param categ_dict: dictionary of class labels, key is integer from categories (the key is referred to as class-integer in documentation)
:param categ_img_set_A, categ_img_set_B: list of integers, corresponding to the two sets of classes
:param categ_indices_img_set_A, categ_indices_img_set_B: list of numpy arrays, list index refers to class, numpy array contains indices of all corresponding images from the pics variable
:param categ_indices_img_set_A___COPY: duplicate with faulty values added, used to demonstrate that the data partition function will detect error
:param pairs_img_set_A: nested numpy arrays, first index is pairing, second is for the two images in pairing, then contains the image data
:param labels_img_set_A: array of 1's and 0's to label corresponding pair from pairs_img_set_A as positive or negative
:param pairs_indiv_categ_img_set_A: numpy arrays, first index corresponds to image pairs from pairs_img_set_A, then contains the integers representing the classes of the pair of images, used to confirm pairings
:param x_test_img_set_B, y_test_img_set_B: pairs and labels of second set of classes, used only for testing
:param x_train_img_set_A, x_test_img_set_A, y_train_img_set_A, y_test_img_set_A: training pairs, testing pairs, training labels and testing labels of first set of classes
:param train_val_threshold: how many pairs to use for training out of the x_train_img_set_A and x_train_img_set_B
:param train_Pi, train_Pj: numpy arrays of input pairs, separated, for building model
:param train_y: numpy arrays of positive of negative integer label, corresponding to train_Pi, train_Pj
:param val_Pi, val_Pj: : numpy arrays of input pairs, separated, for validation during model training
:param val_y: numpy arrays of positive of negative integer label, corresponding to val_Pi, val_Pj
:param x_test_joint_img_set, y_test_joint_img_set: pairs and labels of combined data from the 20% testing subset of img_set_A and all of img_set_B
:param define_model_arch_list_input: list of names of functins to generate model architecture
:param batch_sizes_input: list of integers
:param epochs_list_input: list of integers
:param save_plots_input: boolean
:param output_folder_path_input: string containing the desired ouput folder for plots and data files
:param save_models_input: boolean
:param exit_after_one_model_input: boolean
:param history_dict: hold training and testing accuracy and loss
:param train_img_set_A_metrics, test_img_set_A_metrics, test_img_set_B_metrics, test_img_set_joint_metrics: list of floats, loss and accuracy of trained model
"""

# region functions
def euclidean_distance(vects):
	"""
	Calculates the distance between two vectors
	:param vects is as a list of two tensors
	NB this function was mostly copied from the Keras documentation website, https://keras.io
	:return: tensor of distance
	"""
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
	'''
	Backend requirement for Keras
	NB this function was mostly copied from the Keras documentation website, https://keras.io
	:param shapes: list of
	:return: tuple of 1-D output vector shape
	'''
	shape1, shape2 = shapes
	return (shape1[0], 1)


def contrastive_loss(y_true, distance):
	"""
	Converts the distance measure and postive/negative pair value into the cost.
	:param y_true indicates if the pairing are same class (1) or not (0).
	:param distance is a tensor of Euclidean distance between the two
	:return: contrastive loss cost as a tensor
	NB that positive pairs are marked 1, negative pairs as 0.
	NB this function was mostly copied from the Keras documentation website.
	"""
	margin = 1
	square_distance = K.square(distance)
	margin_square = K.square(K.maximum(margin - distance, 0))
	return 0.5 * (y_true * square_distance + (1 - y_true) * margin_square)

def test_contrastive_loss(y_true, cost_text_x, cost_test_y):
	"""
	Test the cost functions code (euclidean_distance and contrastive_loss). Replicates the calculations in numpy, then prints out the calculated value from the cost functions and from numpy. It tests with both a positive pair and a negative pair.
	:param y_true indicates if the pairing are same class (1) or not (0).
	:return: nothing, just prints out the cost from the two different calculations
	NB that positive pairs are marked 1, negative pairs as 0
	"""
	x = np.array(cost_text_x)  # vector to use as example
	xtf = tf.convert_to_tensor(x, dtype=float)  # convert to tensor
	xtf = tf.reshape(xtf, [-1, 9])  # reshape tensor to match Siamese net input

	y = np.array(cost_test_y)
	ytf = tf.convert_to_tensor(y, dtype=float)
	ytf = tf.reshape(ytf, [-1, 9])

	y_pred_func = euclidean_distance([xtf, ytf])  # calculate distance using functions
	y_pred_np = np.sqrt(sum(np.square(x - y)))  # calculat distance using numpy
	func_value = contrastive_loss(y_true, y_pred_func) # calculate cost with functions

	# calculate cost with numpy
	margin = 1
	square_pred_np = np.square(y_pred_np)
	margin_square_np = np.square(max(margin - y_pred_np, 0))
	np_output = round(0.5 * (y_true * square_pred_np + (1 - y_true) * margin_square_np), 5) # rounded for readability

	with tf.Session() as sess:
		func_output = float(func_value.eval())  # convert function outputs to float
	func_output = round(func_output, 5) # for readability

	if y_true == 1:
		print("This tests a positive pairing:")
	else:
		print("This tests a negative pairing:")
	print("The contrastive loss cost calculated by the euclidean_distance and contrastive_loss functions is: ",
	      func_output)
	print("The contrastive loss cost calculated with numpy is: ", np_output)
	print("The difference between the outputs is: ", abs(np_output-func_output))


def accuracy(y_true, y_pred):
	"""
	Calculate accuracy during training
	:param y_true indicates if the pairing are same class (1) or not (0).
	:param y_pred is estimated distance between two vectors
	:return: the average accuracy of all predictions in this batch
	"""
	# NB the arbitrary cut off point for the threshold between negative and positive label predictions was set as 0.5
	return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def confirm_data_partition(partition, category_dictionary, category_subset_values):
	"""
	Confirms the partition of the original data into the two sets of classes from assignment instructions. Compares the category-integer for each picture to the list of category-integers for the set of classes
	:param partition: list of numpy arrays, list index refers to class, numpy array contains indices of corresponding image from the pics variable
	:param category_dictionary: dictionary of class labels, key is integer from categories
	:param category_subset_values: numpy arrays, first index corresponds to image pairs from partition, then contains the integers representing the classes of the pair of images, for code testing
	:return: nothing, just prints out the results of testing the code
	"""
	correct_partition = True
	for i in range(len(partition)): # for each category
		for j in range(len(partition[i])): # for each index in the category
			if np.all(category_subset_values != categories[partition[i][j]]): # check that the corresponding class-integer is from the set of clases
				print("This picture is in the wrong subset!\nCategory: {}\nIndices: [{}, {}]".format(
					categ_dict[partition[i][j]], i, j))
				correct_partition = False
	if correct_partition:
		print("The partition is correct.\nThis subset contains only: ", end=" ")
		for x in category_subset_values:
			print(category_dictionary[int(x)], end=", ")
	else:
		print("\nThe data partition is incorrect!!")


def create_pairs(pics_for_pairs, indices, categories_for_pairs):
	"""
	Creates one positive and one negative pairing for each image in the data set
	:param pics_for_pairs: numpy array of image data
	:param indices: of pics_for_pairs of images in this set of classes, indices[a][b] hold the pic index of the b'th image from the class corresponding to the a'th category from pics_for_pairs
	:param categories_for_pairs: corresponding category
	:return: pairs: numpy array of image data pairings
	:return: labels: numpy array of positive/negative indicator for pairs
	:return: pairs_indiv_categ: numpy arrays, first index corresponds to image pairs from , then contains the integers representing the classes of the pair of images, for code testing
	"""
	pairs = []
	labels = []
	pairs_indiv_categ = []
	num_classes = len(indices)
	n = min([len(indices[d]) for d in range(num_classes)]) - 1
	for class_index in range(num_classes): # for each class in the set of classes
		for i in range(n):  # for every image from this class
			# positive pair
			Pi, Pj = indices[class_index][i], indices[class_index][i + 1] # find the indices for two neighbouring images from this class (neighbours from the class but not necessarily from the dataset3)
			pairs += [[pics_for_pairs[Pi], pics_for_pairs[Pj]]] # add the image data for this pair
			pairs_indiv_categ += [[categories_for_pairs[Pi], categories_for_pairs[Pj]]] # add the two classes to a separate variable for code testing

			# negative pair
			inc = random.randrange(1, num_classes) # random number, less than number of classes
			dn = (class_index + inc) % num_classes # because inc can't be zero or the number of classes, dn must be a different class
			Pi, Pj = indices[class_index][i], indices[dn][i] # indices of negative pairing
			pairs += [[pics_for_pairs[Pi], pics_for_pairs[Pj]]] # append the image data for the two negative pair images
			pairs_indiv_categ += [[categories_for_pairs[Pi], categories_for_pairs[Pj]]] # add the two classes to a separate variable for code testing
			labels += [1,0] # append the labels positive and negative

	return np.array(pairs), np.array(labels), np.array(pairs_indiv_categ)

def visual_test_random_pairs(num_to_view, pairs, labels, pairs_indiv_categ, output_folder_path_input):
	"""
	For testing code. Displays image pairs, along with their class and the corresponding positive/negative label for visual evaluation by the user
	:param num_to_view: how many images to display
	:param pairs: pairs of image data
	:param labels: labels
	:param pairs_indiv_categ: pairs of image data, to test code by identifying image class from a separate method to the combination of pairs and labels
	:param output_folder_path_input: to save images to
	:return: nothing, displays images with text
	"""
	for _ in range(num_to_view):
		pic_index = random.randint(0, len(pairs)-1) # select a random image from pairs
		categ_true = labels[pic_index] # get positive/negative label

		if categ_true == 1:
			truth = "Should be SAME"
		else:
			truth = "Should be DIFFERENT"

		fig=plt.figure()
		fig.add_subplot(1, 2, 1)
		plt.imshow(pairs[pic_index, 0, :, :, 0])
		plt.title(categ_dict[int(pairs_indiv_categ[pic_index][0])]) # display the class-integer for one image from the pair, separate data source than the pairs and labels, for code testing
		fig.add_subplot(1, 2, 2)
		plt.imshow(pairs[pic_index, 1, :, :, 0])
		plt.title(categ_dict[int(pairs_indiv_categ[pic_index][1])]) # display the class-integer for one image from the pair, separate data source than the pairs and labels, for code testing
		fig.suptitle('Visual Test of Pairs Labelling\n\n {}'.format(truth))
		filename = "visual_test_pair_" + str(pic_index)
		fig_filename = output_folder_path_input + filename + ".png"
		plt.savefig(fig_filename)


def numeric_test_random_pairs(pics_original, num_to_test, pairs, labels):
	"""
	Confirms accuracy of positive/negative labels for randomly selected pairings. For each image in pairing, it pulls out a row of data from the middle of the image, then searches the original dataset for a match. When found, it finds the corresponding class-integer. Then it compares the two class-integers with each other, and then with the corresponding label for the pair. If the two class-integers are the same, then the label should be 1; if the two class-integers are different, then the label should be 0. The function prints out a confirmation of this, or states that there is a faulty label.
	:param pics_original, original data
	:param num_to_test: how many images to test
	:param pairs: pairs of image data
	:param labels: corresponding positive/negative label
	:return: nothing, just prints
	"""
	def first_subarray(full_array, sub_array):
		"""
		Searches the original image data for a vector of image data as an alternate way of finding the images class-integer.
		:param full_array: entire original image dataset, flattened to a single vector
		:param sub_array: a 28-length-vector taken from the 15th row of an image
		:return: the corresponding class-integer
		"""
		n = len(full_array)
		k = len(sub_array)
		# the following line compares the sub-array to the 15th row of each image, then 'matches' holds the index of the matching vector wrt the array of images
		matches = np.argwhere([np.all(full_array[start_ix:start_ix + k] == sub_array)
		                       for start_ix in range(391, n - 1, 784)])
		return categories[int(matches[0])]

	picsFlat = pics_original.ravel() # full image dataset
	pairsFlat = pairs.ravel() # pairs image dataset

	for _ in range(num_to_test):
		test = random.randint(0, len(pairs) - 1) # randomly select an image pairing

		# for the first image in the pairing:
		start_i, end_i = (test * 2 * 784 + 391), (test * 2 * 784 + 391 + 28) # find the indices of the first and last pixel, of the 15th row,
		target_i = pairsFlat[start_i: end_i] # pull out the target vector of pixels
		# repeat for the second image in the pairing:
		start_j, end_j = (test * 2 * 784 + 784 + 391), (test * 2 * 784 + 784 + 391 + 28)
		target_j = pairsFlat[start_j: end_j]

		categ_i = first_subarray(picsFlat, target_i) # returns the category of the first image
		categ_j = first_subarray(picsFlat, target_j) # returns the category of the second image
		categ_true = labels[test] # get the positive/negative label of the pairing

		if categ_true == 1:
			truth = "Yes"
		else:
			truth = "No"

		print("\n\nPair selected:", test)

		# are the two categories found from the pixel search the same and is it supposed to be a positive pairing
		if (categ_i == categ_j) and (categ_true == 1):
			print("\t\t\t\tThis pairing is CORRECT")
		# are the two categories found from the pixel search different and is it supposed to be a negative pairing
		elif (categ_i != categ_j) and (categ_true == 0):
			print("\t\t\tThis pairing is CORRECT")
		else: # otherwise there was an error with the pairs creation
			print("\t\t\tThis pairing is WRONG")

		# pring summary of outcome for this pairing
		print("First picture category: {}\nSecond picture category: {}\nShould they be the same: {}".format(
			categ_dict[int(categ_i)], categ_dict[int(categ_j)], truth))

def train_models(define_model_arch_list_input, batch_sizes_input, epochs_list_input, output_folder_path_input, save_models_input, exit_after_one_model_input):
	"""
	This function trains the models. It contains booleans to save plots and data. It contains a boolean to exit after building one model for demonstration purposes
	:param define_model_arch_list_input: list of names of functins to generate model architecture
	:param batch_sizes_input: list of integers
	:param epochs_list_input: list of integers
	:param output_folder_path_input: string, containing desired output folder for plots and data files
	:param save_models_input: boolean
	:param exit_after_one_model_input: boolean
	:return:
	"""
	for define_model_architecture in define_model_arch_list_input:
		for batch_size in batch_sizes_input:
			for epochs in epochs_list_input:
				start = time.time()
				# network definition
				base_network = define_model_architecture(input_shape)
				input_i = Input(shape=input_shape)
				input_j = Input(shape=input_shape)
				processed_i = base_network(input_i)
				processed_j = base_network(input_j)
				# assign output function to a variable
				distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_i, processed_j])
				# assign inputs and output
				model = Model([input_i, input_j], distance)
				# build model
				model.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
				history = model.fit(
					[train_Pi, train_Pj], train_y,
				          batch_size=batch_size,
				          epochs=epochs,
					validation_data=([val_Pi, val_Pj],val_y))

				runtime = time.time() - start
				print("Model training took: ", runtime)

				history_dict = history.history
				# compute loss and accuracy on training and test datasets using the finished model
				print("The metrics returned are {} and {}.".format(model.metrics_names[0], model.metrics_names[1]))

				test_img_set_A_metrics = model.evaluate([x_test_img_set_A[:, 0], x_test_img_set_A[:, 1]], y_test_img_set_A,
				                                        verbose=0)

				test_img_set_B_metrics = model.evaluate([x_test_img_set_B[:, 0], x_test_img_set_B[:, 1]], y_test_img_set_B,
				                                        verbose=0)

				test_img_set_joint_metrics = model.evaluate([x_test_joint_img_set[:, 0], x_test_joint_img_set[:, 1]],
				                                            y_test_joint_img_set, verbose=0)

				# Plot training & accuracy values
				plt.figure()
				plt.plot(history_dict['accuracy'])
				plt.plot(history_dict['val_accuracy'])
				plt.subplots_adjust(top=0.7)
				# from the model name, extract the size (small or large) and whether it uses regularizers
				model_arch_name = "_".join(define_model_architecture.__name__.split('_')[-3:])
				plt.annotate('Model Architecture:\n {}\nBatch size: {}'.format(model_arch_name, batch_size), xy=(0.5, 0.2), xycoords='axes fraction')
				plt.title('Model accuracy\n\nTest set A accuracy:   {:.2%}\nTest set B accuracy:   {:.2%}\nJoint test set accuracy:   {:.2%}\n'.format(test_img_set_A_metrics[1], test_img_set_B_metrics[1], test_img_set_joint_metrics[1]))
				plt.ylabel('Accuracy')
				plt.xlabel('Epoch')
				plt.legend(['Train', 'Test'], loc='upper left')
				filename = "accuracy_architecture_" + model_arch_name + "_BS_" + str(batch_size) + "_epochs_" + str(epochs)
				fig_filename = output_folder_path_input + "model_" + filename + ".png"
				plt.savefig(fig_filename)


				plt.figure()
				plt.plot(history_dict['loss'])
				plt.plot(history_dict['val_loss'])
				plt.subplots_adjust(top=0.7)
				plt.annotate('Model Architecture:\n {}\nBatch size: {}'.format(model_arch_name, batch_size), xy=(0.5, 0.8), xycoords='axes fraction')
				plt.title('Model loss\n\nTest set A loss:   {:.2%}\nTest set B loss:   {:.2%}\nJoint test set loss:   {:.2%}\n'.format(test_img_set_A_metrics[0], test_img_set_B_metrics[0], test_img_set_joint_metrics[0]))
				plt.ylabel('Loss')
				plt.xlabel('Epoch')
				plt.legend(['Train', 'Test'], loc='upper left')
				filename = "loss_architecture_" + model_arch_name + "_BS_" + str(batch_size) + "_epochs_" + str(epochs)
				fig_filename = output_folder_path_input + "model_" + filename + ".png"
				plt.savefig(fig_filename)

				if save_models_input:
					history_filename = output_folder_path_input + "model_history_" + filename + ".json"
					model_filename = output_folder_path_input + "model_" + filename + ".json"
					model_weights_filename = output_folder_path_input + "model_weights_" + filename + ".h5"
					model_pairs_and_labels_filename = output_folder_path_input + "model_pairs_and_labels_" + filename + ".npz"
					# save history
					json.dump(history_dict, open(history_filename, 'w'))
					# # save model
					model_json = model.to_json()
					with open(model_filename, "w") as json_file:
					    json_file.write(model_json)
					# # save weights
					model.save_weights(model_weights_filename)
					# # save pairs and labels
					np.savez(model_pairs_and_labels_filename,
					         x_train_img_set_A=x_train_img_set_A,
					         x_test_img_set_A=x_test_img_set_A,
					         y_train_img_set_A=y_train_img_set_A,
					         y_test_img_set_A=y_test_img_set_A,
					         x_test_joint_img_set=x_test_joint_img_set,
					         y_test_joint_img_set=y_test_joint_img_set)
				if exit_after_one_model_input:
					return

				# to make it easier to look at the console output mid-training and identify where in loops of model-architecture, batch size and number of epochs we're up to
				print("***********************************************************************")
				print("***********************************************************************")
				print("Just finished: batchSize: ", batch_size,
				      "\nEpochs: ", epochs,
				      "\nModel architecture " + model_arch_name)
				print("***********************************************************************")
				print("***********************************************************************")

# endregion functions

# region pre-processing
# Combine all the data, convert images from /256 to /1, reshape to keras input requirements
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
pics = np.concatenate((x_train, x_test))
categories = np.concatenate((y_train, y_test))
del(x_train, x_test, y_train, y_test)

pics, categories = shuffle(pics, categories, random_state = 0) # random shuffle in case there was a pattern to the original data distribution
pics = pics.astype('float32')
pics /= 255
img_rows, img_cols = pics.shape[1:3]
pics = pics.reshape(pics.shape[0], img_rows, img_cols, 1) # change shape from (70000, 28, 28) to (70000, 28, 28, 1) to fit the Keras-CNN input requirements
categories = categories.reshape(categories.shape[0], 1) # (70000,) to (70000,1)

# Create dictionary with numeric key for category as string
categ_dict = {
0 :	"top",
1 :	"Trouser",
2 :	"Pullover",
3 :	"Dress",
4 :	"Coat",
5 :	"Sandal",
6 :	"Shirt",
7 :	"Sneaker",
8 :	"Bag",
9 :	"Ankle boot"}

# create lists of the categories for the two subsets
categ_img_set_A= [0, 1, 2, 4, 5, 9] # ["top", "trouser", "pullover", "coat", "sandal", "ankle boot"]
categ_img_set_B = [3, 7, 8, 6] # ["dress", "sneaker", "bag", "shirt"]


# partition data into the two required category-subsets
categ_indices_img_set_A = [np.where(categories == i)[0] for i in categ_img_set_A]
categ_indices_img_set_B = [np.where(categories == i)[0] for i in categ_img_set_B]

# create pairs of image inputs and positive/negative labels for the two subsets
pairs_img_set_A, labels_img_set_A, pairs_indiv_categ_img_set_A = create_pairs(pics, categ_indices_img_set_A, categories)
x_test_img_set_B, y_test_img_set_B, pairs_indiv_categ_img_set_B = create_pairs(pics, categ_indices_img_set_B, categories)
# endregion preprocessing

# region test code
# test code visually
visual_test_random_pairs(4, pairs_img_set_A, labels_img_set_A, pairs_indiv_categ_img_set_A, output_folder_path)
visual_test_random_pairs(4, x_test_img_set_B, y_test_img_set_B, pairs_indiv_categ_img_set_B, output_folder_path)

# test code with pixel search
numeric_test_random_pairs(pics, 5, pairs_img_set_A, labels_img_set_A)
numeric_test_random_pairs(pics, 5, x_test_img_set_B, y_test_img_set_B)

# Test contrastive loss functions against numpy implementation
x = [1.1, 2,   3,    4.02, 5.2, 6,    7,   8.077, 9]
y = [1,   2.2, 3.06, 4,    5,   6.03, 7.2, 8,     9.2]
test_contrastive_loss(1, x, y) # positive pair
test_contrastive_loss(0, x, y) # negative pair
x = [0.01,  0.05, -0.07, -0.023, 2.6,  0.012, 0.548,  0.79, -.2]
y = [0.01, -0.05, -0.047, 0.023, 2.4, -0.02,  0.0548, 0.7,  -.2]
test_contrastive_loss(1, x, y) # positive pair
test_contrastive_loss(0, x, y) # negative pair

# test partition
confirm_data_partition(categ_indices_img_set_A, categ_dict, categ_img_set_A)
confirm_data_partition(categ_indices_img_set_B, categ_dict, categ_img_set_B)

# confirm an error would be detected by confirm_data_partition, create intentionally wrong data to input
categ_indices_img_set_A___COPY = categ_indices_img_set_A.copy()
categ_indices_img_set_A___COPY.append(np.array([2,3]))
confirm_data_partition(categ_indices_img_set_A___COPY, categ_dict, categ_img_set_A)
print("NB the line above refers to an intentional error, used to check that the confirm_data_partition would find a fault in the partition.")
del categ_indices_img_set_A___COPY
# endregion test code

# separate 20% of image set A for testing after model is built
x_train_img_set_A, x_test_img_set_A, y_train_img_set_A, y_test_img_set_A = train_test_split(pairs_img_set_A, labels_img_set_A, test_size=.2)
# join image set B with the 20% of image set A for joint testing
x_test_joint_img_set = np.concatenate((x_test_img_set_A, x_test_img_set_B), axis=0)
y_test_joint_img_set = np.concatenate((y_test_img_set_A, y_test_img_set_B), axis=0)

# region different model architectures
input_shape = pairs_img_set_A.shape[2:]

# define four models with different sizes for the layers, ##########but the same structure otherwise
def define_model_architecture_small_no_regularizers(input_shape):
	input = Input(shape=input_shape)

	# convolutional layer 1
	x = Conv2D(16, (3, 3), input_shape=input_shape, padding='same', activation='relu')(input)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# convolutional layer 1
	x = Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten
	x = Flatten()(x)
	x = Dense(64, activation='relu')(x)
	x = Dropout(0.25)(x)
	x = Dense(32, activation='softmax')(x)
	return Model(input, x)

def define_model_architecture_large_no_regularizers(input_shape):
	input = Input(shape=input_shape)
	# convolutional layer 1
	x = Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu')(input)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# convolutional layer 1
	x = Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.25)(x)
	x = Dense(64, activation='softmax')(x)
	return Model(input, x)

def define_model_architecture_small_with_regularizers(input_shape):
	input = Input(shape=input_shape)

	# convolutional layer 1
	x = Conv2D(16, (3, 3), input_shape=input_shape, padding='same', activation='relu')(input)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# convolutional layer 1
	x = Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten
	x = Flatten()(x)
	x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	x = Dropout(0.25)(x)
	x = Dense(32, activation='softmax')(x)
	return Model(input, x)

def define_model_architecture_large_with_regularizers(input_shape):
	input = Input(shape=input_shape)
	# convolutional layer 1
	x = Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu')(input)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# convolutional layer 1
	x = Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten
	x = Flatten()(x)
	x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	x = Dropout(0.25)(x)
	x = Dense(64, activation='softmax')(x)
	return Model(input, x)

# endregion different model architectures

# region split training data

# subset the training data into training and validation groups
train_val_threshold = 60000
train_Pi = x_train_img_set_A[:train_val_threshold, 0]
train_Pj = x_train_img_set_A[:train_val_threshold, 1]
train_y = y_train_img_set_A[:train_val_threshold]

val_Pi = x_train_img_set_A[train_val_threshold:, 0]
val_Pj = x_train_img_set_A[train_val_threshold:, 1]
val_y = y_train_img_set_A[train_val_threshold:]
# endregion

# region model function inputs
define_model_arch_list = [define_model_architecture_small_no_regularizers,
                          define_model_architecture_large_no_regularizers,
                          define_model_architecture_small_with_regularizers,
                          define_model_architecture_large_with_regularizers]
batch_sizes = [512, 1024, 2048, 4096]
epochs_list = [2,3,4,5]
save_plots = False
save_models = False
exit_after_one_model = False
# endregion model function inputs

# build models
train_models(define_model_arch_list, batch_sizes, epochs_list, output_folder_path, save_models_input = False, exit_after_one_model_input = True)
print("Total runtime: {} seconds".format((time.time()-totalStart).__round__()))