import re
import numpy as np
import glob
from keras.models import model_from_json
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

TEST_DIC = "CarData/TestImages/"
TEST_SPLIT_DIC = "result/test_split/"


class CNNTest:
    GOOD_LEN_TS = 5
    GOOD_BEST_TS = 3
    GOOD_TS = 0.75
    STEP = 10
    WINDOW_W = 100
    WINDOW_H = 40
    TEST_DAMAGED_FILES = ["test-1", "test-10", "test-108"]

    test_data = []
    test_filename = []

    test_splits = []
    test_points = []

    def __init__(self, file_id):
        json_file = open("model/save/model_" + str(file_id) + ".json", 'r')
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        self.model.load_weights("model/save/model_" + str(file_id) + ".h5")

    def load_test_data(self):
        for f in glob.glob(TEST_DIC + "*.pgm"):
            self.test_data.append(self.read_pgm(f, byteorder='<'))
            self.test_filename.append(f.split("/")[2].split(".")[0])

    def load_test_split_data(self, test_file):
        for f in glob.glob(TEST_SPLIT_DIC + test_file + "*.pgm"):
            self.test_splits.append(self.read_pgm(f, byteorder='<'))
            f_name = f.split("/")[2].split(".")[0]
            point = (int(f_name.split("_")[1]), int(f_name.split("_")[2]))
            self.test_points.append(point)

    def get_good_points(self):
        good = []
        for i in range(len(self.test_splits)):
            test_split = np.array(self.test_splits[i]).reshape(-1, 100, 40, 1)
            test_split = test_split.astype('float32')
            test_split = test_split / 255.
            predicted_class = self.model.predict_classes(test_split)
            predicted_prob = self.model.predict(test_split)[0][0]

            if predicted_class == 0:
                good.append([predicted_prob, self.test_points[i]])

        if len(good) == 0:
            return None
        good = sorted(good, key=itemgetter(0), reverse=True)

        good = good[:self.GOOD_LEN_TS]
        while len(good) > self.GOOD_BEST_TS:
            if good[len(good) - 1][0] < self.GOOD_TS:
                good.remove(good[len(good) - 1])
            else:
                break

        boxes = []
        detected = []
        c = 0
        for i in range(len(good)):
            if good[i] in detected:
                continue
            boxes.append([])
            boxes[c].append(good[i])
            for j in range(i + 1, len(good)):
                if good[j] in detected:
                    continue
                if abs(good[i][1][0] - good[j][1][0]) <= self.WINDOW_W / 2 and abs(
                        good[i][1][1] - good[j][1][1]) <= self.WINDOW_H / 4:
                    detected.append(good[j])
                    boxes[c].append(good[j])
            detected.append(good[i])
            c += 1

        boxes = boxes[:2]
        good_boxes = []
        for box in boxes:
            s = 0
            div = 0
            for i in range(len(box)):
                s += box[i][1][0] * (len(box) - i)
                div += len(box) - i
            good_boxes.append([int(s / div), box[0][1][1]])
        return good_boxes

    def show_img_with_boxes(self, img, points):
        _, ax = plt.subplots(1)
        ax.imshow(img)
        for point in points:
            ax.add_patch(
                patches.Rectangle((point[0], point[1]), self.WINDOW_W, self.WINDOW_H, linewidth=1, edgecolor='r',
                                  facecolor='none'))
        plt.show(block=True)

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
