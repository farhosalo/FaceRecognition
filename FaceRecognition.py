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
trainDatasetRaw = learnData.take(trainSize)
testDatasetRaw = learnData.skip(trainSize)
validDatasetRaw = testDatasetRaw.skip(testSize)
testDatasetRaw = testDatasetRaw.take(testSize)

print(
    "nElements={0}, Training={1}, testSize={2} and validSize={3}".format(
        nElement, len(trainDatasetRaw), len(testDatasetRaw), len(validDatasetRaw)
    )
)

# Create a preprocessing pipeline
preprocess = tf.keras.Sequential(
    [
        tf.keras.layers.Resizing(
            height=common.HEIGHT, width=common.WIDTH, crop_to_aspect_ratio=True
        ),
        tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input),
    ]
)

# Apply preprocessing pipeline to datasets
trainDataset = trainDatasetRaw.map(lambda X, Y: (preprocess(X), Y))
trainDataset = trainDataset.prefetch(1)

testDataset = testDatasetRaw.map(lambda X, Y: (preprocess(X), Y))

validDataset = validDatasetRaw.map(lambda X, Y: (preprocess(X), Y))


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
inputs = tf.keras.layers.Input(shape=common.IMAGE_SHAPE, name="NewInput")

# Connect layers
x = dataAugmentation(inputs)
x = baseModel(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Add output layer
outputs = tf.keras.layers.Dense(nClasses, activation="softmax")(x)

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
history = model.fit(trainDataset, validation_data=validDataset, epochs=common.EPOCHS)

stop = time.time()

# Calculate training duration print it out
print(f"Training took: {(stop-start)/60} minutes")

# After we achieve good weights, we can make the base model layers trainable again
# and continue training with lower learning rates

for layer in baseModel.layers:
    layer.trainable = True
learningRate = 0.01
momentum = 0.9
optimizer = tf.keras.optimizers.legacy.SGD(
    learning_rate=learningRate, momentum=momentum
)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

start = time.time()
history = model.fit(trainDataset, validation_data=validDataset, epochs=common.EPOCHS)
stop = time.time()
print(f"Training took: {(stop-start)/60} minutes")


# Save the model
workingDir = os.path.dirname(__file__)
modelPath = os.path.join(workingDir, common.MODEL_NAME)
model.save(modelPath)
