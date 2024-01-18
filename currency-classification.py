import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 

import os
from pathlib import Path

import tensorflow as tf


##**Train and validation dataset preparation**

train_dir = '/content/drive/MyDrive/Indian-Currency-Classification-main/train'   # Address of your training dataset from drive 

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
          train_dir, 
          labels='inferred', 
          label_mode='int',
          class_names=['10', '20', '50', '100', '200', '500', '2000'], 
          color_mode='rgb', 
          batch_size=32, 
          image_size=(128, 128), 
          shuffle=True, 
          seed=123, 
          validation_split=0.2, 
          subset='training',
          interpolation='bicubic', 
          follow_links=False, 
          smart_resize=True
        )

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
          train_dir, 
          labels='inferred', 
          label_mode='int',
          class_names=['10', '20', '50', '100', '200', '500', '2000'], 
          color_mode='rgb', 
          batch_size=32, 
          image_size=(128, 128), 
          shuffle=False, 
          seed=123, 
          validation_split=0.2, 
          subset='validation',
          interpolation='bicubic', 
          follow_links=False, 
          smart_resize=True
        )
def normalize(image,label):
  """
    Returns normalized image and its label
  """
  image = tf.cast(image/255. ,tf.float32)
  return image,label

# Normalizing dataset for better accuracy

train_dataset = train_dataset.map(normalize)
validation_dataset = validation_dataset.map(normalize)


##**Model Architecture**


model = tf.keras.Sequential([
            
            tf.keras.layers.Conv2D(64, (1, 1), input_shape=(128, 128, 3)),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Activation('relu'),
            
            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(16, (3, 3)),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)), 

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(7),
            tf.keras.layers.Activation('softmax')
          ])

print(model.summary())

##**Model Optimization Definition**

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
    metrics=['accuracy']
    )

##**Training the Model**

history=model.fit(train_dataset,epochs=50)

plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='b')
plt.plot(history.history['loss'], color='r')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['accuracy', 'loss'])

plt.show()


##**Extracting test images for results**

val_loss, val_acc = model.evaluate(validation_dataset, verbose='auto')
print(f"Validation:\n\tloss:{val_loss} \n\taccuracy:{val_acc}")

test_dir = '/content/drive/MyDrive/Indian-Currency-Classification-main/test'

test_imageID = []
# r=root, d=directories, f = files

for r, d, f in os.walk(test_dir, topdown=True):
    for file in f:
      if '.jpg' in file:
          test_imageID.append(Path(file).stem)
test_imageID = sorted(test_imageID)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
          test_dir, 
          labels=None, 
          label_mode=None,
          class_names = None,
          color_mode='rgb', 
          batch_size=1,
          shuffle = False,
          image_size=(128, 128), 
          interpolation='bicubic', 
          smart_resize=True
        )

def normalize_test(image):
  """
    Returns normalized image and its label
  """
  image = tf.cast(image/255. ,tf.float32)
  return image

# # Normalizing dataset
test_dataset = test_dataset.map(normalize_test)

predictions = model.predict(test_dataset)
pred_category = np.argmax(predictions,axis = 1)     # Extracting index of the label with maximum probability
labels_name = ['10', '20', '50', '100', '200', '500', '2000']
pred_output_labels = [labels_name[i] for i in pred_category]

##**Visualizing the output predictions**

test_images = list(test_dataset.as_numpy_iterator()) # Returns an iterable over the elements of the dataset, with their tensors converted to numpy arrays
num_test_images = len(test_images)

subplot_rows = num_test_images//6 + (1 if num_test_images%6!=0 else 0)
subplot_columns = num_test_images if num_test_images<6 else 6

plt.figure(figsize=(20, 20))
i = 0     # Iterator

for images in test_images:
  ax = plt.subplot(subplot_rows, subplot_columns, i + 1)
  plt.imshow((np.squeeze(images) * 255).astype(np.uint8), cmap = 'gray')
  plt.title(pred_output_labels[i])
  plt.axis("off")
  i = i + 1

plt.tight_layout()
plt.show()