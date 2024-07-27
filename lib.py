from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import cv2 as cv
import os

random.seed(1009)
np.random.seed(1009)



