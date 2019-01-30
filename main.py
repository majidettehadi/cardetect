import re
import numpy as np
import glob
from keras.models import model_from_json
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

TEST_DIC = "CarData/TestImages/"
TEST_SPLIT_DIC = "result/test_split/"

GOOD_LEN_TS = 5
GOOD_BEST_TS = 3
GOOD_TS = 0.75
STEP = 10
WINDOW_W = 100
WINDOW_H = 40
TEST_DAMAGED_FILES = ["test-1", "test-10", "test-108"]

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


def load_test_data():
    test_loaded = []
    test_f_name_loaded = []
    for f in glob.glob(TEST_DIC + "*.pgm"):
        test_loaded.append(read_pgm(f, byteorder='<'))
        test_f_name_loaded.append(f.split("/")[2].split(".")[0])
    return test_loaded, test_f_name_loaded


def load_test_split_data(test_file):
    test_split_loaded = []
    test_split_point_loaded = []
    for f in glob.glob(TEST_SPLIT_DIC + test_file + "*.pgm"):
        test_split_loaded.append(read_pgm(f, byteorder='<'))
        f_name = f.split("/")[2].split(".")[0]
        point = (int(f_name.split("_")[1]), int(f_name.split("_")[2]))
        test_split_point_loaded.append(point)
    return test_split_loaded, test_split_point_loaded


def get_good_points(trained_model, test_splits, test_points):
    good = []
    for i in range(len(test_splits)):
        test_split = np.array(test_splits[i]).reshape(-1, 100, 40, 1)
        test_split = test_split.astype('float32')
        test_split = test_split / 255.
        predicted_class = trained_model.predict_classes(test_split)
        predicted_prob = trained_model.predict(test_split)[0][0]

        if predicted_class == 0:
            good.append([predicted_prob, test_points[i]])

    if len(good) == 0:
        return None
    good = sorted(good, key=itemgetter(0), reverse=True)

    good = good[:GOOD_LEN_TS]
    while len(good) > GOOD_BEST_TS:
        if good[len(good) - 1][0] < GOOD_TS:
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
            if abs(good[i][1][0] - good[j][1][0]) <= WINDOW_W / 2 and abs(
                    good[i][1][1] - good[j][1][1]) <= WINDOW_H / 4:
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
            s += box[i][1][0] * (len(box)-i)
            div += len(box)-i
        good_boxes.append([int(s/div), box[0][1][1]])
    return good_boxes


def show_img_with_boxes(img, points):
    _, ax = plt.subplots(1)
    ax.imshow(img)
    for point in points:
        ax.add_patch(patches.Rectangle((point[0], point[1]), WINDOW_W, WINDOW_H, linewidth=1, edgecolor='r',
                                       facecolor='none'))
    plt.show(block=True)


if __name__ == "__main__":
    json_file = open(MODEL_FILE, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(MODEL_WEIGHT_FILE)

    test_data, test_f_name = load_test_data()
    no_points_number = 0
    with open("result/foundLocations.txt", 'w') as result:
        for i_d in range(len(test_data)):
            line = test_f_name[i_d].split("-")[1] + ":"
            if test_f_name[i_d] in TEST_DAMAGED_FILES:
                line = line + " (" + str(int(len(test_data[i_d][0]) / 2)) + ","
                line = line + str(int(len(test_data[i_d]) / 2)) + ")"
                result.write(line + "\n")
                continue
            test_split_data, test_split_point = load_test_split_data(test_f_name[i_d])
            found_points = get_good_points(model, test_split_data, test_split_point)
            if found_points is not None:
                for found_point in found_points:
                    line = line + " (" + str(found_point[1]) + ","
                    line = line + str(found_point[0]) + ")"
                # show_img_with_boxes(test_data[i_d], found_points)
            else:
                no_points_number += 1
                line = line + " (" + str(int(len(test_data[i_d][0]) / 2)) + ","
                line = line + str(int(len(test_data[i_d]) / 2)) + ")"
                # print("No box for " + test_f_name[i_d] + " :(")
            result.write(line + "\n")

    print("Number of files couldn't load: " + str(len(TEST_DAMAGED_FILES)) + "\nNumber of files no box found: " +
          str(no_points_number))
