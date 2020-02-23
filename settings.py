import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_NAME = "images"
DATA_DIR_PATH = os.path.join(ROOT_DIR, r"data\{0}".format(DATA_DIR_NAME))
TRAIN_DIR_PATH = os.path.join(DATA_DIR_PATH, "train")
VAL_DIR_PATH = os.path.join(DATA_DIR_PATH, "validation")
TEST_DIR_PATH = os.path.join(DATA_DIR_PATH, "test")

print(DATA_DIR_PATH)
print(TRAIN_DIR_PATH)
print(VAL_DIR_PATH)
print(TEST_DIR_PATH)
print(ROOT_DIR)