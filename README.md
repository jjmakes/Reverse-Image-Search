# CS370-002-Project

Team Member Contacts:

- John Makely jm672@njit.edu **(AWS CONTACT)**
- Thomas Lanzetti tl362@njit.edu
- Andrew Kritzler ak2426@njit.edu

[LFW dataset](http://vis-www.cs.umass.edu/lfw/)

## Installation

Install the Python dependencies using pip.

```sh
python3 -m pip install -r requirements.txt
```

## Prototype

### Preparation

Images in the `samples` directory will be converted to haarcascaded and cropped numpy arrays using `generate.py` and stored into `dataset`

### Training

`train.py` will prepare the contents of `dataset` for KNN classification. The script outputs `model.npy` (training set to be used by KNN) and `names.json` (names of faces with their relative ids)

### Testing

Images placed in the `tests` directory will be classified in `test.py`. The script will output the predicted name of each test image.

TODO:

- tests only recognize Andrew Peirsol [1]. examine the training alg to see if there's any issues. test again if this might be because peirsol had much more training data than the others (initially; extras have been deleted now).
- train against the entire dataset
- make more user friendly (url input? notebook?)
