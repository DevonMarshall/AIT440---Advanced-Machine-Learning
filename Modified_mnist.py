#############################################################################
# coding=utf-8                                                              #
# Copyright 2024 The TensorFlow Datasets Authors.                           #
#                                                                           #
# Licensed under the Apache License, Version 2.0 (the "License");           #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at                                   #
#                                                                           #
#     http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                           #
# Unless required by applicable law or agreed to in writing, software       #
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################
# Additional modification to this file were made by Devon Marshall

"""MNIST, Fashion MNIST, KMNIST and EMNIST."""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

# Data Loading:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Data Normalization:
# 255 was used since it is the largest a pixel can be and will change the range from 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Shaping the data
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Training and validation split
x_train, x_val = x_train[:75000], x_train[25000:]
y_train, y_val = y_train[:75000], y_train[25000:]

# Function to help with training the model
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
    )

datagen.fit(x_train)

# Model creation
model = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),  # Cant place input shape here or it causes warnings
    Dense(128, activation='relu'),
    Dropout(0.2),  # Dropout layer to help prevent overfitting
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(10, activation='softmax') # Anything below 10 causes a crash
])

# Compile the model with:
# - Optimizer: 'adam' (efficient for many cases)
# - Loss: 'sparse_categorical_crossentropy' (suitable for integer labels)
# - Metrics: tracking accuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Establishes batch size, step size, number of epochs
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    steps_per_epoch=len(x_train) // 32,
    epochs=20,
    validation_data=(x_val, y_val)
)

# -------------------------------------------------------------------------
# Testing the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Making predictions
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Creating the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
            yticklabels=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()


# Plotting the predictions
def plot_sample_predictions(x, y_true, y_pred, num_samples=10):
    plt.figure(figsize=(15, 4))
    for i in range(num_samples):
        index = np.random.randint(0, len(x))
        image = x[index].reshape(28, 28)
        true_label = y_true[index]
        pred_label = y_pred[index]

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.suptitle("Sample Predictions")
    plt.show()

plot_sample_predictions(x_test, y_test, y_pred, num_samples=10)

# Visualisation of data
idx = np.random.randint(0, len(x_test))
sample_image = x_test[idx]
sample_prob = y_pred_probs[idx]
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title("Test Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(class_names, sample_prob)
plt.xticks(rotation=45)
plt.ylabel("Probability")
plt.title("Predicted Class Probabilities")
plt.show()
