import numpy as np
import cv2, os, knn

# Parse through a dataset to prepare face analysis
img_dict = {}
face_count = 0

train_dir = 'samples'
data_path = './data/'

for f_name in os.listdir(train_dir):
    for p_img in os.listdir(os.path.join(train_dir, f_name)):
        face_count += 1
        if f_name in img_dict.keys():
            img_dict[f_name].append(os.path.join(train_dir, f_name, p_img))
        else:
            img_dict[f_name] = [os.path.join(train_dir, f_name, p_img)]

# Initialize haar cascade filters provided by
# https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

# Used to pad around extracted faces
offset = 10

# Metrics for counting detected faces
detect_count = 0


# Looping through all image directory values
for person in img_dict:
    # Create a new array for person's images 
    # print(person)
    # face_data = []
    for photo in img_dict[person]:
        # Reading the photo into the img var
        img = cv2.imread(photo)
        cv2.imshow('photo', img)


        # Apply grayscale to the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face Detection
        # Cascading params: image, scaleFactor, minNeighbors
        face = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(face) == 0:
            continue
        detect_count += 1

        x,y,w,h = face[0]

        # Drawing a rectangle around the face coordinates
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 

        # Slicing face from original image
        face_cut = img[y-offset:y+h+offset,x-offset:x+w+offset]

        # Resizing faces to 128x128px
        face_cut = cv2.resize(face_cut,(128,128))
        
        # Display each face and wait for keypress before proceeding
        cv2.imshow('crop', face_cut)
        cv2.waitKey(0)
    # Save the dataset
    # face_data = np.asarray(face_data)
    # face_data = face_data.reshape((face_data.shape[0],-1))
    # np.save(dataset_path + person+ '.npy', face_data)
    # print("Saved " + person + "'s data to /data")

print('Detected ' + str(detect_count/face_count*100) + '%')
print('(' + str(detect_count) + '/' + str(face_count) + ')')

cv2.waitKey(0)
cv2.destroyAllWindows()
