import tensorflow as tf

SEED = 42
HEIGHT = WIDTH = 224
IMAGE_SIZE = (HEIGHT, WIDTH)
IMAGE_SHAPE = IMAGE_SIZE + (3,)
BATCH_SIZE = 10
FACES_DIR = "./faces"
EPOCHS = 15
MODEL_NAME = "face_recognition.h5"
CLASS_NAMES_FILE = "ClassNames.txt"

CLASS_NAMES_FILE = "ClassNames.txt"
MODEL_NAME = "face_recognition.h5"


def split(dataset, trainRatio, validRatio):
    nElement = len(dataset)
    trainDataset = dataset.take(int(nElement * trainRatio))
    validTestDataset = dataset.skip(int(nElement * trainRatio))
    validDataset = validTestDataset.take(int(validRatio * nElement))
    testDataset = validTestDataset.skip(int(validRatio * nElement))
    return trainDataset, validDataset, testDataset
