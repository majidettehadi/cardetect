import sys
import glob
from model import CNNModel

CONFIG_DIC = "model/config/"

start = 0
end = len(glob.glob(CONFIG_DIC + "*"))
if len(sys.argv) == 2:
    start = int(sys.argv[1])
elif len(sys.argv) == 3:
    start = int(sys.argv[1])
    end = int(sys.argv[2])
else:
    print("ERROR 1")
    exit(1)

config_files = glob.glob(CONFIG_DIC + "*")
for i in range(start, end):
    cnn = CNNModel(config_files[i])
    cnn.load_data()
    cnn.prepossessing()
    cnn.gen_valid_data()
    cnn.compile_model()
    cnn.train_model()
    cnn.print_score()
    cnn.save_model()
