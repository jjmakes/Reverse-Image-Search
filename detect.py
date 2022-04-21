import cv2

# Used to pad around extracted faces
offset = 10

# Initialize haar cascade filters provided by
# https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

# Crop faces from photos
def crop(photo_path):
    
    # Read Photo into the img var
    img = cv2.imread(photo_path)
    # cv2.imshow('photo', img)
 
    # Apply grayscale to the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Face Detection
    # Cascading params: image, scaleFactor, minNeighbors
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    if len(face) == 0:
        return face
 
    x,y,w,h = face[0]
 
    # Drawing a rectangle around the face coordinates
    # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
 
    # Slicing face from original image
    face_cut = img[y-offset:y+h+offset,x-offset:x+w+offset]
 
    # Resizing faces to 128x128px
    face_cut = cv2.resize(face_cut,(128,128))

    # Display each face and wait for keypress before proceeding
    # cv2.imshow('crop', face_cut)
    # cv2.waitKey(0)
 
    return face_cut
