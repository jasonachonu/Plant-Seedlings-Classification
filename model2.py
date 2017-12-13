import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from time import time

PATH = "/home/mike/kaggle"
TRAINING_FOLDER = os.path.join(PATH, "train")
TRAIN_MOD = os.path.join(PATH, "train_mod")
TEST_IMAGES = os.path.join(PATH, "test")
TEST_MOD = os.path.join(PATH, "test_mod")
INPUT_SIZE = 50
USE_MOD = True

if USE_MOD:
    X = np.load('serialized/X_mod_{}.npy'.format(INPUT_SIZE))
else:
    X = np.load('serialized/X_{}.npy'.format(INPUT_SIZE))
targets = np.load('serialized/targets.npy')

# split into training and validation sets and stratify to preserve class imbalances
X_train, X_val, y_train, y_val = train_test_split(X, targets, test_size=0.2, random_state=42, stratify=targets)


def conv_net(optimizer='sgd', summary=True):
    conv_net = Sequential()
    conv_net.add(Conv2D(filters=50, kernel_size=(5, 5),
                        activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    conv_net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    conv_net.add(Conv2D(filters=100, kernel_size=(5, 5),
                        activation='relu'))
    conv_net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    conv_net.add(Dropout(rate=0.25))
    conv_net.add(Flatten())
    conv_net.add(Dense(500, activation='relu'))
    conv_net.add(Dense(12, activation='softmax'))

    if summary:
        conv_net.summary()

    if optimizer == 'sgd':
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.90, nesterov=False)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    else:
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    conv_net.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return conv_net


# set seed for reproducibility
np.random.seed(100)

# instantiate the model
model = conv_net(optimizer='sgd')
# set up Tensorboard logs if they do not exist
if not os.path.exists(os.path.join(PATH, "logs")):
    os.mkdir(path=os.path.join(os.path.join(PATH, "logs")))
# configure Tensorboard settings
tensorboard = TensorBoard(log_dir=os.path.join(PATH, "logs/{}".format(time())),
                          histogram_freq=1, batch_size=32, write_graph=True)


start = time()

history = model.fit(X_train, y_train, batch_size=32, epochs=20,
                    validation_data=(X_val, y_val),
                    callbacks=[tensorboard])

end = time()

print("Total Training Time: {0:.2f} seconds".format(end - start))

# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy by Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss by Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# save visualization of network architecture
plot_model(model, to_file='model2.png', show_layer_names=True, show_shapes=True)
