import numpy as np
import os, json

dataset_path = 'dataset/'

face_data = []
labels = []

class_id = 0 # Labels for the given file
names = {} # Mapping id & name

# Data Prep
for fx in os.listdir(dataset_path):

    if fx.endswith('.npy'):

        # Create class_id & name mapping
        names[class_id] = fx[:-4] # index removes .npy from name
        print('Loaded ' + fx)

        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # Create Labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)

# Saving label names
json = json.dumps(names)
f = open('names.json', 'w')
f.write(json)
f.close()

# Saving model
np.save('model.npy', trainset)

print("Model finished training and saved to model.npy.\nShape:\n")

print(trainset.shape)
