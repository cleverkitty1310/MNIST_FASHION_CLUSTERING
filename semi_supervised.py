import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
import random
import os

n_labels = 1000
n_train = 5000
(x_train, y_train), (x_test, y_test) = load_data()


x_train = x_train[:n_train].reshape((n_train, 784))
y_train = y_train[:n_train]

x_test = x_test.reshape((10000, 784))

num_labels = len(np.unique(y_train))
print(x_train.shape, num_labels)

x_train_with_label = x_train[:n_labels]
y_train_with_label = y_train[:n_labels]

x_train_no_label = x_train[n_labels:]
y_train_no_label = np.ones_like(y_train[n_labels:]) * -1.

print(y_train_no_label)

model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=0)

clf = model.fit(x_train_with_label, y_train_with_label)

accuracy_score_B = model.score(x_test, y_test)
print('Accuracy Score: ', accuracy_score_B)

x_train = np.concatenate((x_train_with_label, x_train_no_label), axis = 0)
y_train = np.concatenate((y_train_with_label, y_train_no_label), axis = 0)

model_svc = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=0)

self_training_model = SelfTrainingClassifier(base_estimator = model_svc, threshold=0.75, criterion='threshold', max_iter=3, verbose=True)

clf_ST = self_training_model.fit(x_train, y_train)

accuracy_score_ST = clf_ST.score(x_test, y_test)

print('Accuracy Score: ', accuracy_score_ST)