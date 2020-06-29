"""
Created on Mon Jun 29 16:55:33 2020

@author: prbpedro
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
import os
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__)

data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                  fname='flower_photos', untar=True)

data_dir
for x in os.walk(data_dir):
  print(x)

import glob
image_count=len(list(glob.glob(data_dir + '/*/*.jpg')))
image_count

class_names=np.array([item.replace(data_dir + '/', '') for item in glob.glob(data_dir + '/*') if 'LICENSE.txt' not in item])
class_names

import cv2  
from google.colab.patches import cv2_imshow 
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

X = []
y = []


for class_name in class_names:
  i = 0
  images_class_path = list(glob.glob(data_dir + '/' + class_name + '/*.jpg'))
  for image_path in images_class_path:
    image = Image.open(str(image_path))
    label=class_name
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128,128))
    X.append(img)
    y.append(str(label))
  i+=1

X=np.array(X)
X=X/255

le=LabelEncoder()
y=le.fit_transform(y)
y=to_categorical(y,5)

X.shape, y.shape

X[0]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from keras.layers import Conv2D 
from keras.layers import MaxPooling2D

model = keras.Sequential([
  keras.layers.Conv2D(32, (3, 3), input_shape=(128,128,3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(input_shape=(128,128, 3)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(5, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data = (X_test,y_test))

history_dict = history.history
print(history_dict.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

predictions = model.predict(X_test)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.xticks([])
  plt.yticks([])
  img_float32 = np.float32(img)
  plt.imshow(cv2.cvtColor(img_float32, cv2.COLOR_BGR2RGB))
  plt.grid(False)

  predicted_label = np.argmax(predictions_array)

  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label],
                                       100*np.max(predictions_array),
                                       class_names[true_label]),
             color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.xticks([x for x in range(5)])
  plt.yticks([0, predictions_array.max()])
  plt.grid(False)
  bar_plot = plt.bar(range(5), predictions_array, color='#777777')
  plt.ylim([0,1])
  predicted_label = np.argmax(predictions_array)
  bar_plot[predicted_label].set_color('red')
  bar_plot[true_label].set_color('blue')

true_label = []
for b in y_test:
  for z in range(len(b)):
    if b[z] == 1:
      true_label.append(z)

print(true_label)

plt.figure(figsize=(2*2*5, 2*3))
for i in range(10):
  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plot_image(i, predictions, true_label, X_test)
  plt.subplot(1,2,2)
  plot_value_array(i, predictions, true_label)
  plt.tight_layout()
  plt.show()