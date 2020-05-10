import csv
from scipy import ndimage
import cv2
import numpy as np
import os
import sklearn
import math
import matplotlib.pyplot as plt
import random


samples = []
#with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
with open('/root/Desktop/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        samples.append(row)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
'''
all_data = []
# Create histograms of labels for the data set
for sample in samples:
    all_data.append(float(sample[3]))
    
no_zero = []
no_zero[:] = [x for x in all_data if x > 0.01 or x < -0.01]

sample_hist, sample_edges = np.histogram(no_zero, bins=100)

# Plot histograms
plt.bar(sample_edges[:-1], sample_hist, width = 0.01)
plt.title('Training Data')
plt.xlim(min(sample_edges), max(sample_edges))
plt.show()

'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                correction = 0.2 # tuned parameter
                steering_center = float(row[3])
                if (steering_center > 0.01 or steering_center < -0.01): # or random.random() < 0.1):
                    # create adjusted steering measurements for the side camera images
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction

                    # read in images from center, left, and right cameras
                    #path = '/opt/carnd_p3/data/'
                    path = ''
                    img_center = ndimage.imread(path + row[0].strip())
                    img_left = ndimage.imread(path + row[1].strip())
                    img_right = ndimage.imread(path + row[2].strip())

                    images.extend((img_center, img_left, img_right))
                    angles.extend((steering_center, steering_left, steering_right))
                
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle * -1)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set batch size - but will be multiplied by 6 to generate more data within the generator
# Includes a normal and flipped version of the left, center, and right images with adjusted steering angles
batch_size = 1024

# Use the generator function to create the shuffled data one batch at a time
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D

# Added dropout layers to prevent overfitting after each convolution
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100)) #used to be 100
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/(batch_size)), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/(batch_size)), epochs=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')