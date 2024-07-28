import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from modules import *

X = np.load("X.npz.npy")
y = np.load("y.npz.npy")
# faceloader= FaceLoader('./Hau-Face')
# X, y = faceloader.load_classes()

print(X.shape)
print(y.shape)

# dataset = tf.data.Dataset.from_tensor_slices((X, y))

# IMG_SIZE = 160

# def augment(image, label):
#     image = tf.cast(image, tf.float32)
#     image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
#     image = image / 255.0
#     image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
#     image = tf.image.random_brightness(image, max_delta=0.1)
#     return image, label

# X_train = [x / 255. for x in X]
# y_train = [y_ for y_ in y]

# for i in range(3):
#     train_ds = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     train_ds = train_ds.shuffle(1000).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

#     for image, label in dataset.as_numpy_iterator():
#         X_train.append(image)
#         y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# print(X_train.shape)
# print(y_train.shape)
faceaug = FaceAugmentation()
X_train, y_train = faceaug.fit(X, y)
print(X_train.shape)
print(y_train.shape)

facereg = FaceReg()

facereg.training(X_train, y_train)

random_name = os.listdir("./Hau-Face")[random.randint(0, len(os.listdir("./Hau-Face")) - 1)]
random_file = os.listdir(f"./Hau-Face/{random_name}")[random.randint(0, len(os.listdir(f"./Hau-Face/{random_name}")) - 1)]
t_im = cv.imread(f"./Hau-Face/{random_name}/{random_file}")
print(f"./Hau-Face/{random_name}/{random_file}")
print(t_im)
print(facereg.predict(t_im))

print("Applied Augmentation")
print(f"Acc: {facereg.accuracy()}")

faceloader= FaceLoader('./Hau-Face-Test')
X, y = faceloader.load_classes()

X, y = facereg.preprocess(X, y)
X, y = shuffle(X, y, random_state=1009)

print(f"Val acc: {facereg.val_accuracy(X, y)}")
