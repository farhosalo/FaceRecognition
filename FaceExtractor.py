import os
import cv2
from glob import glob
import urllib.request
import CommonDefinitions as common

# Get working directory
workingDir = os.path.dirname(__file__)

# Directory where extracted faces are stored
facesDir = os.path.join(workingDir, common.FACES_DIR)

# Create face directory if it doesn't exist
if not os.path.exists(facesDir):
    os.makedirs(facesDir)

# Directory where original images are stored
imagesDir = os.path.join(workingDir, "images")

# Haarcascade model for face detection
haarCascadeFile = os.path.join(workingDir, "haarcascade_frontalface_default.xml")

# Url from which the Haarcascade model is downloaded
haarCascadeUrl = "https://raw.githubusercontent.com/kipr/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# If the Haarcascade model is not downloaded then download it
if not os.path.isfile(haarCascadeFile):
    urllib.request.urlretrieve(haarCascadeUrl, haarCascadeFile)


faceCascade = cv2.CascadeClassifier(haarCascadeFile)

extensions = ["jpeg", "png", "jpg", "bmp"]
for extension in extensions:
    filenameList = glob(
        os.path.join(imagesDir + "/**", "*.%s" % (extension)), recursive=True
    )
    for filename in filenameList:
        imageFileName = os.path.basename(filename)

        # Get the sub directory of the image file
        subDir = os.path.dirname(os.path.relpath(filename, imagesDir))

        destSubDir = os.path.join(facesDir, subDir)

        # Create sub directory in destination directory "faces/" if not already exists
        if not os.path.exists(destSubDir):
            os.makedirs(destSubDir)

        print(imageFileName)
        image = cv2.imread(filename)

        # Color order when reading with opencv is BGR, so we need to convert it to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detecting faces
        detectedFaces = faceCascade.detectMultiScale(
            rgb,
            scaleFactor=1.3,
            minNeighbors=5,
        )

        # Save all detected faces to destination directory
        index = 0
        for x, y, w, h in detectedFaces:
            index = index + 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            newImage = image[y : y + h, x : x + w]
            faceName = "%s_Face_%02d.jpg" % (imageFileName, index)
            cv2.imwrite(os.path.join(destSubDir, faceName), newImage)
