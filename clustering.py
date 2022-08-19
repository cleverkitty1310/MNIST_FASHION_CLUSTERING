import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets.mnist import load_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d

import os

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))

num_labels = len(np.unique(y_train))
print(x_train.shape, num_labels)

bw_train = np.zeros((60000, 784))
for index, value in enumerate(x_train):
    value = np.array(value, dtype='float32') #Threshold method only works on 8-bit integer/32 bit floating point arrays
    ret, bw = cv2.threshold(value, 127, 255, cv2.THRESH_BINARY)
    bw = bw.reshape(784)
    bw_train[index] = bw

bw_test = np.zeros((10000, 784))
for index, value in enumerate(x_test):
    value = np.array(value, dtype='float32') #Threshold method only works on 8-bit integer/32 bit floating point arrays
    ret, bw = cv2.threshold(value, 127, 255, cv2.THRESH_BINARY)
    bw = bw.reshape(784)
    bw_test[index] = bw

scaler = StandardScaler()
scaler.fit(bw_train)

bw_train_no_pca = scaler.transform(bw_train)
bw_test_no_pca = scaler.transform(bw_test)

pca = PCA(.95)

pca.fit(bw_train_no_pca)

print(pca.n_components_)

bw_train_pca = pca.transform(bw_train_no_pca)
bw_test_pca = pca.transform(bw_test_no_pca)
# bw_train_pca_invese = pca.inverse_transform(bw_train_pca)

# kmeans_no_pca = KMeans(init = "k-means++", n_clusters = 10, n_init = 35)
# kmeans_no_pca.fit(bw_train_no_pca)

kmeans_pca = KMeans(init = "k-means++", n_clusters = 10, n_init = 35)
kmeans_pca.fit(bw_train_pca)

num_cluster_labels_pca = len(np.unique(kmeans_pca.labels_))
cluster_indexes = [[] for i in range(num_labels)]
for i, label in enumerate(kmeans_pca.labels_):
    cluster_indexes[label].append(i)

true_indexes = [[] for i in range(num_labels)]
for i, label in enumerate(y_train):
    true_indexes[label].append(i)

print('With PCA')
for i in range(num_labels):
    print('No. of items in Cluster ' + str(i) + ': ' + str(len(cluster_indexes[i])))

trace =[[] for i in range(0, 10)]
colors = ['Reds','Greens' ,'Blues','Purples','pink','magma','twilight','Oranges','Greys','copper']

# For 3D plot

fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(0, num_labels):
    my_members = (cluster_indexes[i])
    num_cluster = i
    xdata = bw_train_pca[my_members, 0]
    ydata = bw_train_pca[my_members, 1]
    zdata = bw_train_pca[my_members, 2]    
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap = colors[i])
plt.show()

# For 2D plot
# colors = np.array(["Red","Green","Blue", "Yellow", "Purple", "Cyan", "Black", "Orange", "Grey", "Pink"])
# plt.subplot(1, 2, 1)
# for i in range(0, num_labels):
#     my_members = (true_indexes[i])
#     xdata = bw_train_pca[my_members, 0]
#     ydata = bw_train_pca[my_members, 1]
#     plt.scatter(xdata, ydata, c = colors[i], s = 50)

# plt.subplot(1, 2, 2)
# for i in range(0, num_labels):
#     my_members = (cluster_indexes[i])
#     xdata = bw_train_pca[my_members, 0]
#     ydata = bw_train_pca[my_members, 1]
#     plt.scatter(xdata, ydata, c = colors[i], s = 50)
# plt.show()

test_clusters = kmeans_pca.predict(bw_test_pca)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, test_clusters)

print(acc)