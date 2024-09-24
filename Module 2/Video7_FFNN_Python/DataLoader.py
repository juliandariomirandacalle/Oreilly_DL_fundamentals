import numpy as np
from MNIST import MNIST_Reader

"""
Created by Julian D. Miranda for O'Reilly Media
"""

class MNIST_DataLoader():
    def __init__(self, labels_input = [i for i in range(0,10)]):
        """
        Constructor function.
        Inputs:
            - labels_input: list of target labels to be imported (digits from 0 to 9).
        Outputs:
            - None
        """
        self.labels_input = labels_input
    
    def shuffle_values(self, input_array, lbls):
        """
        Shuffles the observations and labeles randomly pairwise.
        Inputs:
            - input_array: input array to be shuffled
            - lbls: target values to be shuffled
        Outputs:
            - input_array: input array shuffled.
            - lbls: target values shuffled.
        """
        shuffld_idxs = [i for i in range(input_array.shape[2])]
        np.random.seed(42)
        np.random.shuffle(shuffld_idxs)
        input_array = input_array[:,:,shuffld_idxs]
        input_array = np.array([[input_array[:,:,i]] for i in range(input_array.shape[2])])
        lbls = np.array(lbls)[shuffld_idxs]
        return input_array, lbls
    
    def get_images(self,):
        """
        Imports all training and testing images given the input targets labels_input.
        Inputs:
            - None
        Outputs:
            - training_set: training set samples as a multi-dimensional array.
            - testing_set: testing set samples as a multi-dimensional array.
            - training_labels: training labels as a one-dimensional aray.
            - testing_labels testing labels as a one-dimensional aray.
        """
        training_set = np.zeros((28,28,1))
        testing_set = np.zeros((28,28,1))
        training_labels = []
        testing_labels = []
    
        for label in self.labels_input:
            training_set_label, testing_set_label = MNIST_Reader().load_images_from_digit(digit=label)
            training_set = np.append(training_set, training_set_label/255.0, axis=2)
            testing_set = np.append(testing_set, training_set_label/255.0, axis=2)
            training_labels += [label]*training_set_label.shape[2]
            testing_labels += [label]*testing_set_label.shape[2]
        training_set, training_labels = self.shuffle_values(training_set[:,:,1:], training_labels)
        testing_set, testing_labels = self.shuffle_values(testing_set[:,:,1:], testing_labels)
        
        return training_set, testing_set, training_labels, testing_labels

    def split_batches(self, data, labels_data, batch_size = 64):
        """
        Imports all training and testing images given the input targets labels_input.
        Inputs:
            - data: data to be split into homogeneous batches.
            - labels_data: target values to be split into homogeneus batches according to the data split.
            - batch_size: size of the batch of observations.
        Outputs:
            - data_batches: input data split in batches of batch_size length.
            - labels_batches: labeles split in batches of batch_size length.
        """
        n_imgs = data.shape[0]
        n_imgs_batch = n_imgs // batch_size + 1
    
        data_batches = np.array_split(data, n_imgs_batch)
        labels_batches = np.array_split(labels_data, n_imgs_batch)
        return data_batches, labels_batches