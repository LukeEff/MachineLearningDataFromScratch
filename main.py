from random import randint
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.metrics import categorical_crossentropy

train_labels = []
train_samples = []
test_samples = []

for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

for i in range(50):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    random_older = randint(65, 100)
    test_samples.append(random_older)

test_samples = np.array(test_samples)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

# We will use a CPU

"""
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model.summary()
"""


def very_simple_model():
    model = Sequential([
        Dense(32, input_shape=(3,), activation='relu'),
        Dense(2, activation='softmax'),
    ])


def simple_model():
    model = Sequential([
        Dense(16, input_shape=(1,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])

    # Adam is a type of SGD (Stochastic gradient descent) optimizer
    # loss - sparse_categorical_crossentropy is a type of loss function.
    # metrics is just what comes out in the console when training it
    model.compile(Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # function that trains the model.
    # takes training data, the numpy array that holds labels of training data.
    # batch size is how much data to be sent to the data at once.
    # epochs is how many passes of data through the model.
    # shuffle will shuffle the data around with each epoch.
    # verbose specifies how much data we want to see.
    model.fit(scaled_train_samples, train_labels, batch_size=10, epochs=20, shuffle=True, verbose=2)

    # validation split will take a fraction of data for the validation set.
    # alternatively, validation_data = valid_set could be used, where valid_set is some set of data and labels as
    # a list of tuples.
    model.fit(
        scaled_train_samples,
        train_labels,
        validation_split=0.20,
        batch_size=10,
        epochs=20,
        shuffle=True,
        verbose=2)
    return model


def predict(model):
    predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)
    for prediction in predictions:
        print(prediction)


simple_model = simple_model()
predict(simple_model)
