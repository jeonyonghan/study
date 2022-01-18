import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

fashion_mnist = tf.keras.datasets.mnist.load_data()
(train_x, train_y), (test_x, test_y) = fashion_mnist
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

plt.imshow(train_x[5], cmap = 'gray')
plt.show()
print(train_y[:10])

train_scaled = train_x.reshape(-1, 28, 28, 1)/255.0
train_scaled, val_scaled, train_y, val_y = train_test_split(train_scaled, train_y, test_size = 0.2, random_state =42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = (28,28,1)))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(64, kernel_size = 3, activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.summary()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('best-cnn-model.h5',
                                                   save_best_only = True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 2,
                                                     restore_best_weights = True)
history = model.fit(train_scaled, train_y, epochs = 20,
                    validation_data = (val_scaled, val_y),
                    callbacks = [checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

model.evaluate(train_scaled, train_y)
model.evaluate(val_scaled, val_y)

test_scaled = test_x.reshape(-1, 28, 28, 1)/255.0
model.evaluate(test_scaled, test_y)
