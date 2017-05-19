from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from IPython.display import display, Image


pickle_file = 'notMNIST.pickle'

def read_pickle_file(pickle_file):
    with open(pickle_file, 'rb') as f:
        row_data_set = pickle.load(f)
        #keys = []
        #data_set = []
        #print (row_data_set)
        for key, value in row_data_set.items():
            if key == "train_labels":
                train_labels = value
            elif key == "train_dataset":
                train_dataset = value
            elif key == "valid_dataset":
                valid_dataset = value
            elif key == "valid_labels":
                valid_labels = value
            elif key == "test_labels":
                test_labels = value
            elif key == "test_dataset":
                test_dataset = value
            else:
                print("Something is wrong")
                continue
    return train_labels, train_dataset, valid_labels, valid_dataset, test_labels, test_dataset

train_labels, train_dataset, valid_labels, valid_dataset, test_labels, test_dataset = read_pickle_file(pickle_file)

#print(valid_dataset[0])
#print(valid_labels)
def overlap_check(train_labels, train_dataset, other_labels, other_dataset, sample_amount = 1000):
    index_train_list = []
    #index_other_list = []
    #print(other_dataset.shape)
    length_train = len(train_labels)
    for i in range(len(other_labels)):
        other_dataset_i = other_dataset[i, :, :]
        #train_dataset_i = train_dataset[i, :, :]
        #print(other_dataset_i)
        other_dataset_i_rep = np.tile(other_dataset_i,(length_train,1,1))
        #print(other_dataset_i_rep.shape)
        #print(train_dataset.shape)
        diff_matrx = train_dataset - other_dataset_i_rep
        sum_diff = np.sum(np.sum(diff_matrx, axis=1), axis=1)
        zero_check = np.nonzero(sum_diff == 0)[0]
        #print(zero_check)
        if zero_check.size:
            index_train_list.append(zero_check[0])
        if i % 50 == 1:
            print ('Have Checked', i, 'Steps')
        if i == sample_amount:
            break

    print(index_train_list)
    count = len(index_train_list)
    percent_overlap = count/sample_amount
    print(percent_overlap*100)
    print(count)
    return index_train_list

#overlap_check(train_labels, train_dataset, valid_labels, valid_dataset)
def train_logistic_model(num_data_set = 50, num_class = 10):
    scale_train_data = train_dataset[:num_data_set, :, :]
    scale_train_label = train_labels[:num_data_set]
    nsamples, nx, ny = scale_train_data.shape
    d2_train_dataset = scale_train_data.reshape((nsamples,nx*ny))
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(d2_train_dataset, scale_train_label)
    # cross_validation
    num_cv = num_data_set
    scale_valid_data = valid_dataset[:num_cv,:, :]
    scale_valid_label = valid_labels[:num_cv]
        #scale_valid_data_binary = np.zeros((scale_valid_label.shape[0], num_class))
        #scale_valid_data_binary[np.arange(scale_valid_label.shape[0]), scale_valid_label] = 1
    nval, nxv, nyv = scale_valid_data.shape
    d2_val_dataset = scale_valid_data.reshape((nval, nxv*nyv))
    pred = lr.predict(d2_val_dataset)
    sc = lr.score(d2_val_dataset, scale_valid_label)
    return sc, pred

res = []
acc = []
for item in [50, 100, 200, 500, 1000, 5000, 10000]:
    temp_lost, temp_pred = train_logistic_model(num_data_set=item)
    res.append(temp_pred)
    acc.append(temp_lost)
    print('Finished', item)

plt.figure()
plt.plot([1, 2, 3, 4, 5, 6, 7], acc)
plt.show()


