import sys
import glob
from test import CNNTest

MODEL_DIC = "model/save/"

start = 0
end = len(glob.glob(MODEL_DIC + "*.json"))
if len(sys.argv) == 2:
    start = int(sys.argv[1])
elif len(sys.argv) == 3:
    start = int(sys.argv[1])
    end = int(sys.argv[2])
else:
    print("ERROR 1")
    exit(1)

for i in range(start, end):
    cnn = CNNTest(i)
    cnn.load_test_data()

    no_points_number = 0
    with open("result/locations/foundLocations_" + str(i) + ".txt", 'w') as result:
        for i_d in range(len(cnn.test_data)):
            line = cnn.test_filename[i_d].split("-")[1] + ":"
            if cnn.test_filename[i_d] in cnn.TEST_DAMAGED_FILES:
                line = line + " (" + str(int(len(cnn.test_data[i_d][0]) / 2)) + ","
                line = line + str(int(len(cnn.test_data[i_d]) / 2)) + ")"
                result.write(line + "\n")
                continue
            cnn.load_test_split_data(cnn.test_filename[i_d])
            found_points = cnn.get_good_points()
            if found_points is not None:
                for found_point in found_points:
                    line = line + " (" + str(found_point[1]) + ","
                    line = line + str(found_point[0]) + ")"
                # cnn.show_img_with_boxes(test_data[i_d], found_points)
            else:
                no_points_number += 1
                line = line + " (" + str(int(len(cnn.test_data[i_d][0]) / 2)) + ","
                line = line + str(int(len(cnn.test_data[i_d]) / 2)) + ")"
                # print("No box for " + cnn.test_filename[i_d] + " :(")
            result.write(line + "\n")

    print("At " + str(i) + " Number of files couldn't load: " + str(
        len(cnn.TEST_DAMAGED_FILES)) + "\nNumber of files no box found: " + str(no_points_number))
