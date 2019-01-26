from PIL import Image

TEST_DIC = "dataset/test/"
TEST_SPLIT_DIC = "test_split/"
MAX_TEST_FILE = 169


def create_crop(image, step_height, step_width):
    img = Image.open(TEST_DIC + image + ".pgm")
    img_width, img_height = img.size
    k = 0
    for i in range(0, img_height, step_height):
        if i + 20 > img_height:
            break
        for j in range(0, img_width, step_width):
            if j + 50 > img_width:
                break
            box = (j, i, j + 100, i + 40)
            img.crop(box).save(TEST_SPLIT_DIC + image + "_" + str(k) + ".pgm")
            k += 1


for i in range(MAX_TEST_FILE):
    create_crop("test-" + str(i), 10, 10)
