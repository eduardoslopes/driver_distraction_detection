
# coding: utf-8

# In[1]:


import os
import glob
import cv2
import math
import itertools

import numpy as np
from datetime import datetime
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix

from keras import backend
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, Input, Lambda
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file
from keras.applications.vgg16 import VGG16, preprocess_input


# In[2]:


class DriverDistractionHelper:
    
    def __init__(self, img_rows, img_cols, color_type=1):
        x_train, y_train, driver_id, unique_drivers = self.read_and_normalize_train_data(img_rows, img_cols, color_type)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4484, random_state=57, stratify=y_train)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=4484, random_state=57, stratify=y_train)
        
        self.x_train = x_train.transpose(0, 2, 3, 1)
        self.y_train = y_train
        self.x_test = x_test.transpose(0, 2, 3, 1)
        self.y_test = y_test
        self.x_val = x_val.transpose(0, 2, 3, 1)
        self.y_val = y_val
    
    def read_and_normalize_train_data(self, img_rows, img_cols, color_type):
        cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
        if not os.path.isfile(cache_path) or use_cache == 0:
            x_train, y_train, driver_id, unique_drivers = self.load_train(img_rows, img_cols, color_type)
            self.cache_data((x_train, y_train, driver_id, unique_drivers), cache_path)
        else:
            print('Restore train from cache!')
            (x_train, train_target, driver_id, unique_drivers) = restore_data(cache_path)

        x_train = np.array(x_train, dtype=np.uint8)
        y_train = np.array(y_train, dtype=np.uint8)
        x_train = x_train.reshape(x_train.shape[0], color_type, img_rows, img_cols)
        y_train = np_utils.to_categorical(y_train, 10)
        x_train = x_train.astype('float32')
        x_train /= 255
        print('Train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        return x_train, y_train, driver_id, unique_drivers
    
    def read_and_normalize_manual_test_data(self, img_rows, img_cols, color_type=1):
        cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
        if not os.path.isfile(cache_path) or use_cache == 0:
            test_data = self.load_manual_test(img_rows, img_cols, color_type)
            self.cache_data((test_data), cache_path)
        else:
            print('Restore test from cache!')
            (test_data, test_id) = restore_data(cache_path)

        test_data = np.array(test_data, dtype=np.uint8)
        test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
        test_data = test_data.astype('float32')
        test_data /= 255
        print('Test shape:', test_data.shape)
        print(test_data.shape[0], 'test samples')
        return test_data
    
    def load_train(self, img_rows, img_cols, color_type=1):
        X_train = []
        y_train = []
        driver_id = []

        driver_data = self.get_driver_data()

        print('Read train images')
        for j in range(10):
            print('Load folder c{}'.format(j))
            path = os.path.join('datasets', 'driver_distraction', 'input', 'train', 'c' + str(j), '*.jpg')
            files = glob.glob(path)
            for fl in files:
                flbase = os.path.basename(fl)
                img = self.get_image(fl, img_rows, img_cols, color_type)
                X_train.append(img)
                y_train.append(j)
                driver_id.append(driver_data[flbase])

        unique_drivers = sorted(list(set(driver_id)))
        print('Unique drivers: {}'.format(len(unique_drivers)))
        print(unique_drivers)
        return X_train, y_train, driver_id, unique_drivers
    
    def load_manual_test(self, img_rows, img_cols, color_type=1):
        print('Read manual test images')
        path = os.path.join('datasets', 'driver_distraction', 'input', 'test', '*.jpg')
        files = glob.glob(path)
        X_test = []
        total = 0
        for fl in files:
            flbase = os.path.basename(fl)
            img = self.get_image(fl, img_rows, img_cols, color_type)
            X_test.append(img)
            total += 1
        return X_test
    
    def cache_data(self, data, path):
        if os.path.isdir(os.path.dirname(path)):
            file = open(path, 'wb')
            pickle.dump(data, file)
            file.close()
        else:
            print('Directory doesnt exists')
    
    def get_driver_data(self):
        dr = dict()
        path = os.path.join('datasets', 'driver_distraction', 'input', 'driver_imgs_list.csv')
        print('Read drivers data')
        f = open(path, 'r')
        line = f.readline()
        while (1):
            line = f.readline()
            if line == '':
                break
            arr = line.strip().split(',')
            dr[arr[2]] = arr[0]
        f.close()
        return dr
    
    def get_image(self, path, img_rows, img_cols, color_type=1):
        # Load as grayscale
        if color_type == 1:
            img = cv2.imread(path, 0)
        elif color_type == 3:
            img = cv2.imread(path)
        return cv2.resize(img, (img_cols, img_rows))


# In[3]:


def preprocess_image(im):
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    #im = cv2.resize(cv2.imread(path), (224, 224)).astype(np.float32)
    im = (im - vgg_mean)
    return im[:, ::-1] # RGB to BGR

def create_vgg16(img_rows, img_cols, color_type=1, num_classes=None):
    # we initialize the model
    model = Sequential()

    # Conv Block 1
    model.add(Lambda(preprocess_image, input_shape=(img_rows,img_cols,color_type), output_shape=(img_rows,img_cols,color_type)))

    model.add(Conv2D(64, (3, 3), input_shape=(img_rows,img_cols,color_type), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # FC layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))
    
    model.load_weights('datasets/driver_distraction/scripts/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    
    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    for layer in model.layers[:10]:
       layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=10e-5)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'], )

    return model


# In[4]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, filename=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if not filename == None:
        plt.savefig(filename)
        plt.close()


# In[5]:


img_rows = 224
img_cols = 224
color_type = 3

batch_size = 32
nb_epoch = 5
random_state = 51


# In[55]:


ddh = DriverDistractionHelper(img_rows, img_cols, color_type)


# In[ ]:


from collections import Counter
print Counter(np.argmax(ddh.y_test, axis=1))
print Counter(np.argmax(ddh.y_val, axis=1))
print Counter(np.argmax(ddh.y_train, axis=1))


# In[56]:


model = create_vgg16(img_rows, img_cols, color_type, 10)


# In[ ]:


classes = ['safe driving', 'texting - right', 'talking on the phone - right', 'exting - left', 'talking on the phone - left',
          'operating the radio', 'drinking', 'reaching behind', 'hair and makeup', 'talking to passenger']
keys = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']


# In[ ]:


true_y_test = np.argmax(ddh.y_test, axis=1)


# In[ ]:


for i in range(9):
    model.fit(ddh.x_train, ddh.y_train, batch_size=batch_size, epochs=5, verbose=1, validation_data=(ddh.x_val, ddh.y_val))
    scores = model.evaluate(x=ddh.x_test, y=ddh.y_test, batch_size=128)
    print 'Metrics: ', scores
    y_pred_test = model.predict(ddh.x_test, batch_size=128, verbose=1)
    y_pred_test = np.argmax(y_pred_test, axis=1)
    
    cm = confusion_matrix(true_y_test, y_pred_test)
    plot_confusion_matrix(cm, keys, normalize=False, filename='cm_epoch_'+str(i))


# In[ ]:


model.save('datasets/driver_distraction/scripts/models/vgg16_model.h5')
model.save_weights('datasets/driver_distraction/scripts/models/vgg16_model_weights.h5')

