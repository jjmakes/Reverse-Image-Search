# Reverse Visual Search (CS370-002-Project)

- John Makely (jm672@njit.edu)
- Thomas Lanzetti (tl362@njit.edu)
- Andrew Kritzler (ak2426@njit.edu)

[LFW dataset](http://vis-www.cs.umass.edu/lfw/)

## Introduction
In many areas of study and work, we are interested in finding visual similarities. Reverse visual search can be used to find these important visual similarities of many things, such as artifacts, scenery, and even people’s faces. The term person of interest (PoI) is used to indicate that we reverse search on images of people’s faces. In this project, we implement methods and systems to identify a person of interest’s name by their face when the queries are visual (in the form of an image).

## Data and Background
The data we work with for this project is known as Labeled Faces in the Wild (LFW), a database of face photographs designed for studying the problem of facial verification. Containing more than 13,000 images of faces collected from the web, each face is labeled with the name of the person pictured. There are 1,680 people with at least two distinct photos in the data set.

## Reverse Image Search - Baseline

The purpose of KNN is to find similar vectors to a given vector. We tried using the elastiknn Python library to perform KNN queries. After setting up an OpenSearch cluster using Docker, we realized that elastiknn did not support OpenSearch, which is an open-source implementation of ElasticSearch.
The overall visual search architecture model we aimed to follow is as such: a query image is given as input, a CNN helps find feature vectors of the images, and a KNN search is applied to find the resulting name.

The Viola-Jones Face Detection Technique, popularly known as Haar Cascades, is an Object Detection Algorithm that’s used to identify faces in an image or a real time video, and it is found in many areas. This algorithm makes use of edge or line detection features. It works by collecting Haar features, which are basically calculation that are performed on adjacent rectangular areas at a specific location in a detection window. This calculation is performed by summing the pixel intensities in each region, then calculating the differences of the sums. To train, the model was given a lot of positive images consisting of faces, as well as a lot of negative images not consisting of any faces. The model created from this training is available at the OpenCV GitHub repository https://github.com/opencv/opencv/tree/master/data/haarcascades. This repository has the models stored in XML files, which can be read with the OpenCV methods. These include models for face detection, eye detection, upper body and lower body detection, license plate detection, etc.

![Cropping images with haarcascades](https://github.com/jm672/CS370-002-Project/blob/main/results/crop.png)

The file “convoluted.py” applies crop, edge detection, and grayscale filters to an input image. It does this by first applying an edge detection kernel, using the cv2 filter2D method. At first, we used only the horizontal edge detection kernel, but after trying the vertical one instead we noticed that each captured features the other did not. As a result of this, we decided to combine both the horizontal and vertical edge detections by adding them, with the goal of extracting the important features of both. Finally, the cv2 method cvtColor is used to make the image grayscale.

![Image convolution](https://github.com/jm672/CS370-002-Project/blob/main/results/convolution.png)

## Reverse Image Search Improvement
### MTCNN and FaceNet

Functions for acquiring the images, manipulating with cv2, and converting them to numpy arrays are almost identical to the methods of the baseline notebook.

MTCNN utilizes three stages, or "tasks":

1. Resize an image multiple times. Use a CNN called P-Net to scan the images for faces, where it will usually return many false positives.
2. Another CNN, R-Net, is used to obtain precise bounding boxes for these images.
3. Finally, the last CNN, O-Net, performs the final refinement on the bounding boxes with high precision.

After MTCNN crops out the bounding box for a face, the image moves to FaceNet for feature extraction.

FaceNet takes a person's face as input and outputs an "embedding" vector of the most important features.

FaceNet learns with triplet loss, where a randomly selected anchor is compared to positive and negative examples and reorganized accordingly.

### Implementation

The approach taken for comparing the anchor is what's referred to as "one-shot" classification, which is why FaceNet is known as a Siamese Network.

As opposed to traditional classification methods, where a probability distribution is generated for separate classes, one-shot classification returns a similarity score when two images are compared.

In our demo face_net experiment, we compared the faces of our test images to faces in our dataset sample. The scores returned after comparison yeilded fantastic results, where each test passed correctly.

This demo would not prove to be reliable, as comparing the images takes quite a bit of time (~10 seconds). Crunching quick numbers, a database with 10 thousand images would take around 30 hours to find a match.

![Facenet Demo results](https://github.com/jm672/CS370-002-Project/blob/main/results/facenet_demo.png)

Seeing that the face_net would prove to be reliable, we moved forward with training a proper model. After training a FaceNet model against the lfw database, we found fantastic results.


## References and Related Material

https://arxiv.org/abs/1604.02878

https://www.kaggle.com/code/yhuan95/face-recognition-with-facenet/notebook

https://medium.com/analytics-vidhya/face-recognition-using-knn-open-cv-9376e7517c9f

https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08

https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d

https://github.com/opencv/opencv/tree/master/data/haarcascades
