import re
import numpy as np
import glob
from keras.models import model_from_json
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

TEST_DIC = "CarData/TestImages/"
TEST_SPLIT_DIC = "result/test_split/"

GOOD_TS = 0
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
    test_split_box_loaded = []
    for f in glob.glob(TEST_SPLIT_DIC + test_file + "*.pgm"):
        test_split_loaded.append(read_pgm(f, byteorder='<'))
        f_name = f.split("/")[2].split(".")[0]
        box = (f_name.split("_")[1], f_name.split("_")[2], f_name.split("_")[3], f_name.split("_")[4])
        test_split_box_loaded.append(box)
    return test_split_loaded, test_split_box_loaded


def get_good_boxes(trained_model, test_splits, test_boxes):
    good = []
    top_n_max = [0., 0., 0., 0., 0.]
    top_n_good = [None, None, None, None, None]
    for i in range(len(test_splits)):
        test_split = np.array(test_splits[i]).reshape(-1, 100, 40, 1)
        test_split = test_split.astype('float32')
        test_split = test_split / 255.
        predicted_class = trained_model.predict_classes(test_split)
        predicted_prob = trained_model.predict(test_split)[0][0]

        if predicted_class == 0:
            if predicted_prob > 0.99999999:
                good.append(test_boxes[i])
            elif predicted_prob > GOOD_TS:
                if predicted_prob >= top_n_max[0]:
                    for j in range(len(top_n_max)):
                        if j == len(top_n_max) - 1:
                            top_n_max[len(top_n_max) - 1 - j] = predicted_prob
                            top_n_good[len(top_n_max) - 1 - j] = test_boxes[i]
                            break
                        top_n_max[len(top_n_max) - 1 - j] = top_n_max[len(top_n_max) - 2 - j]
                        top_n_good[len(top_n_max) - 1 - j] = top_n_good[len(top_n_max) - 2 - j]
                elif predicted_prob >= top_n_max[1]:
                    for j in range(len(top_n_max)):
                        if j == len(top_n_max) - 2:
                            top_n_max[len(top_n_max) - 1 - j] = predicted_prob
                            top_n_good[len(top_n_max) - 1 - j] = test_boxes[i]
                            break
                        top_n_max[len(top_n_max) - 1 - j] = top_n_max[len(top_n_max) - 2 - j]
                        top_n_good[len(top_n_max) - 1 - j] = top_n_good[len(top_n_max) - 2 - j]
                elif predicted_prob >= top_n_max[2]:
                    if j == len(top_n_max) - 3:
                        top_n_max[len(top_n_max) - 1 - j] = predicted_prob
                        top_n_good[len(top_n_max) - 1 - j] = test_boxes[i]
                        break
                    top_n_max[len(top_n_max) - 1 - j] = top_n_max[len(top_n_max) - 2 - j]
                    top_n_good[len(top_n_max) - 1 - j] = top_n_good[len(top_n_max) - 2 - j]
                elif predicted_prob >= top_n_max[3]:
                    if j == len(top_n_max) - 4:
                        top_n_max[len(top_n_max) - 1 - j] = predicted_prob
                        top_n_good[len(top_n_max) - 1 - j] = test_boxes[i]
                        break
                    top_n_max[len(top_n_max) - 1 - j] = top_n_max[len(top_n_max) - 2 - j]
                    top_n_good[len(top_n_max) - 1 - j] = top_n_good[len(top_n_max) - 2 - j]
                elif predicted_prob >= top_n_max[4]:
                    if j == len(top_n_max) - 5:
                        top_n_max[len(top_n_max) - 1 - j] = predicted_prob
                        top_n_good[len(top_n_max) - 1 - j] = test_boxes[i]
                        break
                    top_n_max[len(top_n_max) - 1 - j] = top_n_max[len(top_n_max) - 2 - j]
                    top_n_good[len(top_n_max) - 1 - j] = top_n_good[len(top_n_max) - 2 - j]

    for top_good in top_n_good:
        if top_good is None:
            continue
        good.append(top_good)

    if len(good) == 0:
        return None

    good = sorted(good, key=itemgetter(0))
    boxes = []
    i = 0
    while i < len(good):
        iw, iww = int(good[i][0]), int(good[i][2])
        sum_width, sum_height = iw, int(good[i][1])
        c = 1
        i += 1

        for j in range(i, len(good)):
            jw, jww = int(good[j][0]), int(good[j][2])
            if iww > jw:
                c += 1
                i += 1
                sum_width += jw
                sum_height += int(good[j][1])
            else:
                break

        boxes.append(
            [int(sum_width / c), int(sum_height / c), int(sum_width / c) + WINDOW_W, int(sum_height / c) + WINDOW_H])

    return boxes


def show_img_with_boxes(img, boxes):
    _, ax = plt.subplots(1)
    ax.imshow(img)
    for box in boxes:
        ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                       facecolor='none'))
    plt.show(block=True)


if __name__ == "__main__":
    json_file = open(MODEL_FILE, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(MODEL_WEIGHT_FILE)

    test_data, test_f_name = load_test_data()
    no_boxes_number = 0
    with open("result/foundLocations.txt", 'w') as result:
        for i_d in range(len(test_data)):
            line = test_f_name[i_d].split("-")[1] + ":"
            if test_f_name[i_d] in TEST_DAMAGED_FILES:
                line = line + " (" + str(int(len(test_data[i_d][0]) / 2)) + ","
                line = line + str(int(len(test_data[i_d]) / 2)) + ")"
                result.write(line + "\n")
                continue
            test_split_data, test_split_box = load_test_split_data(test_f_name[i_d])
            found_boxes = get_good_boxes(model, test_split_data, test_split_box)
            if found_boxes is not None:
                for found_box in found_boxes:
                    line = line + " (" + str(found_box[1] - 5) + ","
                    line = line + str(found_box[0] - 5) + ")"
                # show_img_with_boxes(test_data[i_d], found_boxes)
            else:
                no_boxes_number += 1
                line = line + " (" + str(int(len(test_data[i_d][0]) / 2)) + ","
                line = line + str(int(len(test_data[i_d]) / 2)) + ")"
                # print("No box for " + test_f_name[i_d] + " :(")
            result.write(line + "\n")

    print("Number of files couldn't load: " + str(len(TEST_DAMAGED_FILES)) + "\nNumber of files no box found: " +
          str(no_boxes_number))
