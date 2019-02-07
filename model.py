import re
import glob
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DIC = "CarData/TrainImages/"

EPOCHS = 50
BATCH_SIZE = 64
NUM_CLASSES = 2
ALPHA = 0.05
WINDOW_W = 100
WINDOW_H = 40

MODEL_FILE = "model/model.json"
MODEL_WEIGHT_FILE = "model/model.h5"


def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def load_data():
    train_loaded = []
    train_label_loaded = []
    for f in glob.glob(DIC + "*.pgm"):
        train_loaded.append(read_pgm(f, byteorder='<'))
        if "neg" in f:
            train_label_loaded.append(1)
        else:
            train_label_loaded.append(0)
    return train_loaded, train_label_loaded


train_data, train_label = load_data()

# prepossessing
train_data = np.array(train_data).reshape(-1, WINDOW_W, WINDOW_H, 1)
train_data = train_data.astype('float32')
train_data = train_data / 255.

train_label = np.array(train_label)
train_label_one_hot = to_categorical(train_label)
train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label_one_hot, test_size=0.01,
                                                                    random_state=13)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(WINDOW_W, WINDOW_H, 1), padding='same'))
model.add(LeakyReLU(alpha=ALPHA))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=ALPHA))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=ALPHA))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=ALPHA))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()

train = model.fit(train_data, train_label, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                  validation_data=(valid_data, valid_label))

scores = model.evaluate(train_data, train_label, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

model_json = model.to_json()
with open(MODEL_FILE, "w") as json_file:
    json_file.write(model_json)
model.save_weights(MODEL_WEIGHT_FILE)
print("Saved model to disk")
