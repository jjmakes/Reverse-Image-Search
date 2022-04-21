import numpy as np

# Find distance between two euclidian values
def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

def predict(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get vector and label
        ix = train[i, :-1]
        iy = train[i, -1]

        # Computing the distance from the test point
        d = distance(test, ix)
        dist.append([d,iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
