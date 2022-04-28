import numpy as np
import os, detect, convoluted

# Parse through a dataset to prepare face analysis
img_dict = {}
face_count = 0

train_dir = "samples"
dataset_path = "dataset/"

for f_name in os.listdir(train_dir):
    for p_img in os.listdir(os.path.join(train_dir, f_name)):
        face_count += 1
        if f_name in img_dict.keys():
            img_dict[f_name].append(os.path.join(train_dir, f_name, p_img))
        else:
            img_dict[f_name] = [os.path.join(train_dir, f_name, p_img)]

# Metrics for counting detected faces
detect_count = 0

# Looping through all image directory values
for person in img_dict:

    # Create a new array for person's images
    face_data = []

    for photo in img_dict[person]:

        print(photo)

        face_cut = detect.crop(photo)

        if len(face_cut) == 0:
            continue

        face_filter = convoluted.edgy_af(face_cut)

        detect_count += 1

        # Add face data to array for dataset export
        face_data.append(face_filter)

    # Save the dataset
    if len(face_data) == 0:
        continue

    # Converting data to np array and saving to /data
    face_data = np.asarray(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))
    np.save(dataset_path + person + ".npy", face_data)

    print("Saved " + person + "'s data to /data")

print("Detected " + str(detect_count / face_count * 100) + "%")
print("(" + str(detect_count) + "/" + str(face_count) + ")")
