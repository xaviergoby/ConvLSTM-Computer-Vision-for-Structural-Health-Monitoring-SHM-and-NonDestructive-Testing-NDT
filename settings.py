import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# IMAGES_DIR = os.path.join(DATA_DIR, "images")
IMAGES_DIR = os.path.join(DATA_DIR, "image_datasets")



DATA_DIR_NAME = "images"
# DATA_DIR_NAME = "image_datasets"
DATA_DIR_PATH = os.path.join(ROOT_DIR, r"data\{0}".format(DATA_DIR_NAME))
TRAIN_DIR_PATH = os.path.join(DATA_DIR_PATH, "train")
VAL_DIR_PATH = os.path.join(DATA_DIR_PATH, "validation")
TEST_DIR_PATH = os.path.join(DATA_DIR_PATH, "test")

NUM_CLASSES = 9

NUM_IMGS = 90
IMG_HEIGHT = 247
IMG_WIDTH = 4101
IMG_CHANNELS = 3

MAX_VALID_IMG_WIDTH = 4100

NUM_FRAMES = 164
FRAME_HEIGHT = IMG_HEIGHT
FRAME_WIDTH = 25
FRAME_CHANNELS = IMG_CHANNELS

IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS) # \\TODO Redundant/Unused
FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)

CNN_2D_INPUT_TENSOR_SHAPE = FRAME_SHAPE
# The shape of 4D tensor which stores a sequence of NUM_FRAMES # of frames
# each with a shape of (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
TIME_DISTRIBUTED_MODEL_INPUT_TENSOR_SHAPE = (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)

TEST_DATASET_FRACTION = 0.05