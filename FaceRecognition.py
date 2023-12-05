import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import CommonDefinitions as common

matplotlib.use("TkAgg")


## Preparing datasets
# Load dataset from faces directory
learnData = tf.keras.utils.image_dataset_from_directory(
    directory=common.FACES_DIR,
    shuffle=True,
    batch_size=common.BATCH_SIZE,
    image_size=common.IMAGE_SIZE,
)

# Get class names from dataset
classNames = learnData.class_names
print(classNames)

# Save class names to file to use it when predicting faces
np.savetxt(common.CLASS_NAMES_FILE, classNames, fmt="%s")

# Get number of classes
nClasses = len(classNames)

# Calculate the size of training, testing and validation sets
nElement = len(learnData)
testSize = int(nElement * 0.2)
validSize = int(nElement * 0.1)
trainSize = nElement - testSize - validSize

# Split dataset into training, testing and validation sets
trainDataset = learnData.take(trainSize)
testDataset = learnData.skip(trainSize)
validDataset = testDataset.skip(testSize)
testDatasetR = testDataset.take(testSize)

# Do buffered prefetching to avoid i/o blocking
AUTOTUNE = tf.data.AUTOTUNE
trainDataset = trainDataset.prefetch(buffer_size=AUTOTUNE)
testDataset = testDataset.prefetch(buffer_size=AUTOTUNE)
validDataset = validDataset.prefetch(buffer_size=AUTOTUNE)


print(
    "nElements={0}, Training={1}, testSize={2} and validSize={3}".format(
        nElement, len(trainDataset), len(testDataset), len(validDataset)
    )
)


# Show some faces after preprocessing
def plotImage(index):
    plt.figure(figsize=(10, 10))
    for images, labels in trainDataset.take(index):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(classNames[labels[i]])
            plt.axis("off")
    plt.waitforbuttonpress()
    plt.show()


# plotImage(index=1)

# Load base model with weights
baseModel = tf.keras.applications.xception.Xception(
    weights="imagenet",
    include_top=False,
)
baseModel.trainable = False

# Create augmentation layers
globalAvgPoolingLayer = tf.keras.layers.GlobalAveragePooling2D(
    name="globalAvgPooling2D"
)

# Create augmentation layer
dataAugmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=common.SEED),
        tf.keras.layers.RandomRotation(factor=0.02, seed=common.SEED),
        tf.keras.layers.RandomContrast(factor=0.2, seed=common.SEED),
    ],
    name="dataAugmentation",
)

# Create input layer
inputs = tf.keras.layers.Input(shape=common.IMAGE_SHAPE, name="input")

# Create output layer
outputLayer = tf.keras.layers.Dense(nClasses, activation="softmax", name="output")

# Create preprocessing layer
preprocessLayer = tf.keras.applications.xception.preprocess_input

# Connect layers
x = dataAugmentation(inputs)
x = preprocessLayer(x)
x = baseModel(
    x, training=False
)  # To make sure that the base model is running in inference mode
x = globalAvgPoolingLayer(x)
outputs = outputLayer(x)


# Create a new model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Freeze base model's layers
for layer in baseModel.layers:
    layer.trainable = False


# Model configuration
learningRate = 0.1
momentum = 0.9
optimizer = tf.keras.optimizers.legacy.SGD(
    learning_rate=learningRate, momentum=momentum
)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

# Plot the model and save the plot to a file
tf.keras.utils.plot_model(model=model, to_file="./model.png", show_shapes=True)

import time

start = time.time()

# Start training
phase1History = model.fit(
    trainDataset, validation_data=validDataset, epochs=common.EPOCHS * 2
)
stop = time.time()

# Calculate training duration print it out
firstLearningPhaseDuration = (stop - start) / 60


# Evaluate the model on test data
model.evaluate(testDataset)

# After we achieve good weights, we can make the base model layers trainable again
# and continue training with lower learning rates
# We will make only the above 2/3 layers trainable

noneTrainableLayers = len(baseModel.layers) // 3

for layer in baseModel.layers[noneTrainableLayers:]:
    layer.trainable = True

learningRate = 0.01
momentum = 0.9
optimizer = tf.keras.optimizers.legacy.SGD(
    learning_rate=learningRate, momentum=momentum
)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

start = time.time()
phase2History = model.fit(
    trainDataset, validation_data=validDataset, epochs=common.EPOCHS
)
stop = time.time()
secondLearningPhaseDuration = (stop - start) / 60

# Evaluate the model on test data
model.evaluate(testDataset)

print(f"First learning phase took: {firstLearningPhaseDuration} minutes")
print(f"First learning phase took: {secondLearningPhaseDuration} minutes")


# Save the model
workingDir = os.path.dirname(__file__)
modelPath = os.path.join(workingDir, common.MODEL_NAME)
model.save(modelPath)

# Plotting the training history
plt.subplot(2, 2, 1)
plt.plot(phase1History.history["loss"])
plt.plot(phase1History.history["val_loss"])
plt.title("phase 1 model loss")
plt.ylabel("loss")
plt.legend(["train", "val"])

plt.subplot(2, 2, 2)
plt.plot(phase1History.history["accuracy"])
plt.plot(phase1History.history["val_accuracy"])
plt.title("phase 1 model performance")
plt.ylabel("rmse")
plt.legend(["train", "val"])

plt.subplot(2, 2, 3)
plt.plot(phase2History.history["loss"])
plt.plot(phase2History.history["val_loss"])
plt.title("phase 2 model loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(["train", "val"])

plt.subplot(2, 2, 4)
plt.plot(phase2History.history["accuracy"])
plt.plot(phase2History.history["val_accuracy"])
plt.title("phase 2 model performance")
plt.ylabel("rmse")
plt.xlabel("epochs")
plt.legend(["train", "val"])

plt.show()
