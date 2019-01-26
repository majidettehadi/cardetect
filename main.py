import re
import numpy as np
import glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


TRAIN_DIC = "dataset/train/"
TEST_DIC = "dataset/test/"
TEST_SPLIT_DIC = "test_split/"

EPOCHS = 1
BATCH_SIZE = 64
NUM_CLASSES = 2
ALPHA = 0.1

WINDOW_SIZES = [i for i in range(40, 160, 10)]


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


def get_size(filename, byteorder='>'):
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
    return int(width), int(height)


def load_train_data():
    train_loaded = []
    train_label_loaded = []
    for f in glob.glob(TRAIN_DIC + "/*.pgm"):
        train_loaded.append(read_pgm(f, byteorder='<'))
        if "neg" in f:
            train_label_loaded.append(1)
        else:
            train_label_loaded.append(0)
    return train_loaded, train_label_loaded


def get_best_bounding_box(img, step=10, window_sizes=WINDOW_SIZES):
    best_box_found = None
    best_box_prob = -np.inf

    # loop window sizes: 20x20, 30x30, 40x40...160x160
    for win_size in window_sizes:
        for top in range(0, img.shape[0] - win_size + 1, step):
            for left in range(0, img.shape[1] - win_size + 1, step):
                # compute the (top, left, bottom, right) of the bounding box
                box = (top, left, top + win_size, left + win_size)

                # crop the original image
                cropped_img = img[box[0]:box[2], box[1]:box[3]]

                # predict how likely this cropped image is dog and if higher
                # than best save it


if __name__ == "__main__":
    train_data, train_label = load_train_data()

    # prepossessing
    train_data = np.array(train_data).reshape(-1, 100, 40, 1)
    train_data = train_data.astype('float32')
    train_data = train_data / 255.

    train_label = np.array(train_label)
    train_label_one_hot = to_categorical(train_label)
    train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label_one_hot, test_size=0.2,
                                                                        random_state=13)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(100, 40, 1), padding='same'))
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

