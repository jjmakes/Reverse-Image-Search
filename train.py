import numpy as np
import cv2
import os

# Searching through a provided directory to prepare access for cv2

img_dict = {}

train_dir = 'samples'

for f_name in os.listdir(train_dir):
    for p_img in os.listdir(os.path.join(train_dir, f_name)):
        if f_name in img_dict.keys():
            img_dict[f_name].append(os.path.join(train_dir, f_name, p_img))
        else:
            img_dict[f_name] = [os.path.join(train_dir, f_name, p_img)]



img = cv2.imread(img_dict['Aaron_Peirsol'][0])
g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('image',g_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
