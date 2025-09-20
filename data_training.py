import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

is_init = False
label = []
dictionary = {}
c = 0

# Load dataset with shape validation
for i in os.listdir():
    if i.endswith(".npy") and i != "labels.npy":
        data = np.load(i)

        if data.ndim == 1:
            data = data.reshape(-1, 1)  # Ensure data is at least 2D

        print(f"{i} shape: {data.shape}")  # Optional debug print

        if not is_init:
            is_init = True
            X = data
            y = np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)
        else:
            if data.shape[1] != X.shape[1]:
                print(f"⚠️ Skipping {i} due to shape mismatch. Expected {X.shape[1]}, got {data.shape[1]}")
                continue
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

# Convert labels to numerical values
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")
y = to_categorical(y)  # Convert labels to one-hot encoding

# Shuffle dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_new = X[indices]
y_new = y[indices]

# Define model
input_shape = (X.shape[1],)
ip = Input(shape=input_shape)

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X_new, y_new, epochs=50)

# Save model and labels
model.save("model.keras")
np.save("labels.npy", np.array(label))
