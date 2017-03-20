import csv
import cv2
import numpy as np
import sklearn

lines = []
with open('./Track1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './Track1/IMG/'+batch_sample[i].split('\\')[-1]
                    image = cv2.imread(name)
                    images.append(image)
                correction = 0.1
                angle = float(batch_sample[3])
                angles.append(angle)
                angles.append(angle+correction)
                angles.append(angle-correction)             

            augmented_images, augmented_angles = [], []
            for image,angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

import keras
from keras.models import Sequential, Model
from keras.layers import Flatten,Dense, Lambda, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Cropping2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5,subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5,subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5,subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
'''model.fit(X_train,y_train,validation_split=0.2,shuffle=True)'''
history_object = model.fit_generator(train_generator,
    samples_per_epoch = (len(train_samples)*6), validation_data = validation_generator,
    nb_val_samples = (len(validation_samples)*6), 
    nb_epoch=5, verbose=1)

'''
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
'''
model.save('model-Track1.h5')

