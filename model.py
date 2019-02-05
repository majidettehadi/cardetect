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


class CNNModel:
    NUM_CLASSES = 2
    WINDOW_W = 100
    WINDOW_H = 40

    DROPOUTS = []
    NODES = []

    train_data = []
    train_label = []
    valid_data = []
    valid_label = []

    train = None

    def __init__(self, filename):
        self.model = Sequential()
        with open(filename) as f:
            lines = f.readlines()
            self.MODEL_FILE = "model/save/model_" + lines[0] + ".json"
            self.MODEL_WEIGHT_FILE = "model/save/model_" + lines[0] + ".h5"
            self.EPOCHS = int(lines[1])
            self.BATCH_SIZE = int(lines[2])
            self.ALPHA = float(lines[3])
            self.TEST_SIZE = float(lines[4])
            self.RANDOM_STATE = int(lines[5])
            for dropout in lines[6].split(","):
                self.DROPOUTS.append(float(dropout))
            for node in lines[7].split(","):
                self.NODES.append(int(node))

    def load_data(self):
        for f in glob.glob(DIC + "*.pgm"):
            self.train_data.append(self.read_pgm(f, byteorder='<'))
            if "neg" in f:
                self.train_label.append(1)
            else:
                self.train_label.append(0)

    def prepossessing(self):
        self.train_data = np.array(self.train_data).reshape(-1, self.WINDOW_W, self.WINDOW_H, 1)
        self.train_data = self.train_data.astype('float32')
        self.train_data = self.train_data / 255.
        self.train_label = np.array(self.train_label)

    def gen_valid_data(self):
        train_label_one_hot = to_categorical(self.train_label)
        self.train_data, self.valid_data, self.train_label, self.valid_label = \
            train_test_split(self.train_data,
                             train_label_one_hot,
                             test_size=self.TEST_SIZE,
                             random_state=self.RANDOM_STATE)

    def compile_model(self):
        self.model.add(
            Conv2D(self.NODES[0], kernel_size=(3, 3), activation='linear',
                   input_shape=(self.WINDOW_W, self.WINDOW_H, 1), padding='same'))
        self.model.add(LeakyReLU(alpha=self.ALPHA))
        self.model.add(MaxPooling2D((2, 2), padding='same'))
        self.model.add(Dropout(self.DROPOUTS[0]))
        self.model.add(Conv2D(self.NODES[1], (3, 3), activation='linear', padding='same'))
        self.model.add(LeakyReLU(alpha=self.ALPHA))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(Dropout(self.DROPOUTS[1]))
        self.model.add(Conv2D(self.NODES[2], (3, 3), activation='linear', padding='same'))
        self.model.add(LeakyReLU(alpha=self.ALPHA))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(Dropout(self.DROPOUTS[2]))
        self.model.add(Flatten())
        self.model.add(Dense(self.NODES[3], activation='linear'))
        self.model.add(LeakyReLU(alpha=self.ALPHA))
        self.model.add(Dropout(self.DROPOUTS[3]))
        self.model.add(Dense(self.NUM_CLASSES, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])

    def train_model(self):
        self.train = self.model.fit(self.train_data, self.train_label, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS,
                                    verbose=1, validation_data=(self.valid_data, self.valid_label))

    def print_score(self):
        scores = self.model.evaluate(self.train_data, self.train_label, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def save_model(self):
        with open(self.MODEL_FILE, "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(self.MODEL_WEIGHT_FILE)
        print("Saved model to disk")

    def read_pgm(self, filename, byteorder='>'):
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
