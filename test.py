import numpy as np
import os, knn, detect, cv2, json

test_dir = 'tests'

# Loading the datasets
trainset = np.load('model.npy')
 
f = open('names.json')
names = json.load(f)
f.close()

for f_name in os.listdir(test_dir):
    photo_path = test_dir + '/' + f_name
    face_cut = detect.crop(test_dir + '/' + f_name)

    if len(face_cut) == 0:
        print(f"Face not detected in test image: {photo_path}.\n")
        continue

    # Make face prediction
    out = knn.predict(trainset,face_cut.flatten())
    pred_name = names[str(int(out))];

    print("Image " + photo_path + " is predicted to be: ")
    print(pred_name + "\n")

    cv2.imshow('tests', face_cut)
    cv2.waitKey(0)

cv2.destroyAllWindows()
