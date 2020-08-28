# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:58:04 2020

@author: ivis
"""

from ELM import *
from chest_xray_loader import *
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms
import time
from Time import *


##data loader
"""
train_data, train_label = DataLoader('train')
test_data, test_label = DataLoader('test')

np.save("Pneunomia_train_data.npy", train_data)
np.save("Pneunomia_train_label.npy", train_label)
np.save("Pneunomia_test_data.npy", test_data)
np.save("Pneunomia_test_label.npy", test_label)
"""

train_data = np.load("Pneunomia_train_data.npy")
train_label = np.load("Pneunomia_train_label.npy")
test_data = np.load("Pneunomia_test_data.npy")
test_label = np.load("Pneunomia_test_label.npy")

print("data load complet...")

##ELM parameters
input_size = 3 * 64 * 64
output_size = 2
hidden_size = 50

##create ELM model
model = ELM(input_size, output_size, hidden_size)
print("model complet...")

##train ELM model
start = time.time()

model.train(train_data, train_label.reshape(-1, 1))

print("Time: ", timeSince(start, 1 / 100))

##test ELM model
y_pred = model.predict(test_data)
y_pred = (y_pred > 0.5).astype(int)

##accuracy
print('Accuracy: ', accuracy_score(test_label, y_pred))

