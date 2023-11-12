# Face detection and recognition

## Description

This is a test project for face recognition with Tensorflow using transfer learning.
This project can be applied to an order with personal photos to recognize family members.
The project consists of three parts:

- Face detection and extraction
- Face recognition
- Face prediction

## Face detection and extraction

Here we are using haarcascade method implemented in opencv to detect faces on an image, this is licensed under "Intel License Agreement For Open Source Computer Vision Library": (<https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml>)

After extracting the faces you have to organize them all in a directory structure lik this: <br/><br/>

```
    faces
    |--Person1
    |  |--filename1.jpg
    |  |--filename2.jpg
    |  |--filename3.jpg
    |  |--...
    |--Person2
    |  |--filename1.jpg
    |  |--filename2.jpg
    |  |--filename3.jpg
    |  |-- ...
    |..................
```

## Face prediction

After training and saving the model, we reload the model und use it to predict faces. Just run the Predict.py script and pass it the image path as an argument.
