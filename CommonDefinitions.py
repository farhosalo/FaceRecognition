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

# Create a preprocessing pipeline
preprocess = tf.keras.Sequential(
    [
        tf.keras.layers.Resizing(height=HEIGHT, width=WIDTH, crop_to_aspect_ratio=True),
        tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input),
    ]
)
