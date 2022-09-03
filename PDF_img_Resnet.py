import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import numpy as np
import cv2

training_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=7,
                                        horizontal_flip=True,
                                        zoom_range=0.2)

train_dataset = training_generator.flow_from_directory('C:/Users/ZMO-WIN-GaneshI-01/Downloads/PDF_img/train_set',
                                                        target_size = (180, 180),
                                                        batch_size = 32,
                                                        class_mode = 'categorical',
                                                       shuffle = True)

train_dataset.classes

train_dataset.class_indices

test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_directory('C:/Users/ZMO-WIN-GaneshI-01/Downloads/PDF_img/test_set',
                                                     target_size = (180, 180),
                                                     batch_size = 32,
                                                     class_mode = 'categorical',
                                                     shuffle = False)

model = Sequential()

# adding convolutional layers
model.add(Conv2D(32, (3,3), input_shape = (180,180,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

#hidden layers
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = 128, activation='relu'))

# #hidden layers
# network.add(Dense(units = 3137, activation='relu'))
# network.add(Dense(units = 3137, activation='relu'))

#output layer
model.add(Dense(units = 2, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(train_dataset, batch_size = 32, epochs=10, validation_data=test_dataset)

history.history.keys()

plt.plot(history.history['val_loss']);
plt.plot(history.history['val_accuracy']);

test_dataset.class_indices

# predictions values range between 0 and 1
predictions = model.predict(test_dataset)
predictions

predictions = np.argmax(predictions, axis = 1)
print(predictions)
print(test_dataset.classes)

from sklearn.metrics import accuracy_score
accuracy_score(test_dataset.classes, predictions)

from sklearn.metrics import classification_report
print(classification_report(test_dataset.classes, predictions))

resnet_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=5,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)

resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(2, activation='softmax'))

resnet_model.summary()

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

history = resnet_model.fit(train_dataset, epochs=10, validation_data=test_dataset)

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

# predictions values range between 0 and 1
predictions = resnet_model.predict(test_dataset)
predictions

predictions = np.argmax(predictions, axis = 1)
print(predictions)
print(test_dataset.classes)

from sklearn.metrics import accuracy_score
accuracy_score(test_dataset.classes, predictions)

from sklearn.metrics import classification_report
print(classification_report(test_dataset.classes, predictions))

# After making layer.trainanble True ##
resnet_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=2,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=True

resnet_model.add(pretrained_model)

resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(2, activation='softmax'))

resnet_model.summary()

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

history = resnet_model.fit(train_dataset, epochs=10, validation_data=test_dataset)

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

# predictions values range between 0 and 1
predictions = resnet_model.predict(test_dataset)
predictions

predictions = np.argmax(predictions, axis = 1)
print(predictions)
print(test_dataset.classes)

from sklearn.metrics import accuracy_score
accuracy_score(test_dataset.classes, predictions)

from sklearn.metrics import classification_report
print(classification_report(test_dataset.classes, predictions))