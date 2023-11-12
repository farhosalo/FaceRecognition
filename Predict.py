import tensorflow as tf
import os
import sys
import numpy as np
from PIL import Image
import CommonDefinitions as common

if len(sys.argv) < 2:
    sys.exit("No input specified!")

print(sys.argv[1])


workingDir = os.path.dirname(__file__)
modelPath = os.path.join(workingDir, common.MODEL_NAME)
filePath = os.path.join(workingDir, sys.argv[1])

# Load class names from file
classNames = np.loadtxt(common.CLASS_NAMES_FILE, dtype=str)
print(classNames)


model = tf.keras.models.load_model(modelPath)

# Show the model architecture
model.summary()


img = Image.open(filePath)
img = np.array(img)
preProcessedImage = common.preprocess(img)
yProba = np.argmax(model.predict(preProcessedImage[None, :, :]))
print(classNames[yProba])
