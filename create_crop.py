from PIL import Image

DIC = "CarData/TestImages/"
SPLIT_DIC = "result/test_split/"

MAX_FILES = 169
STEP = 10
WINDOW_W = 100
WINDOW_H = 40


def create_and_save_crops(image, step_height, step_width):
    img = Image.open(DIC + image + ".pgm")
    img_width, img_height = img.size
    for i in range(0, img_height, step_height):
        if i + WINDOW_H - STEP > img_height:
            break
        for j in range(0, img_width, step_width):
            if j + WINDOW_W - STEP > img_width:
                break
            box = (j, i, j + WINDOW_W, i + WINDOW_H)
            img.crop(box).save(
                SPLIT_DIC + image + "_" + str(j) + "_" + str(i) + "_" + str(j + WINDOW_W) + "_" + str(
                    i + WINDOW_H) + ".pgm")


for file_no in range(MAX_FILES):
    create_and_save_crops("test-" + str(file_no), STEP, STEP)
