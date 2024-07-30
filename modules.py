from lib import *

class FaceLoader:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        faces = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                single_face = self.extract_face(path)
                faces.append(single_face)
            except Exception as e:
                print(f"Error loading {im_name}: {e}")
        return faces

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            faces = self.load_faces(path)
            labels = [sub_dir] * len(faces)
            print(f"Loaded successfully: {len(faces)} faces from {sub_dir}")
            self.X.extend(faces)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)

class FaceReg:
    def __init__(self):
        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.model = SVC(kernel='linear', probability=True)
        self.encoder = LabelEncoder()

    def embedding(self, face_img):
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        yhat = self.embedder.embeddings(face_img)
        return yhat[0]

    def preprocess(self, X, y):
        embedded_X = [self.embedding(img) for img in X]
        self.X = np.asarray(embedded_X)
        self.encoder.fit(y)
        self.y = self.encoder.transform(y)
        return self.X, self.y

    def training(self, X, y):
        self.preprocess(X, y)
        self.X, self.y = shuffle(self.X, self.y, random_state=1009)
        self.model.fit(self.X, self.y)
        return self.model

    def predict(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(image)[0]['box']
        image = image[y:y+h, x:x+w]
        image = cv.resize(image, (160, 160))
        emb_img = self.embedding(image)
        y_preds = self.model.predict([emb_img])
        y_preds = self.encoder.inverse_transform(y_preds)
        return y_preds

    def accuracy(self):
        print(self.X.shape)
        y_preds = self.model.predict(self.X)
        return accuracy_score(self.y, y_preds)
    
    def val_accuracy(self, X, y):
        print(X.shape)
        y_preds = self.model.predict(X)
        return accuracy_score(y, y_preds)
    
class FaceAugmentation:

    def __init__(self):
        self.IMG_SIZE = 160

    def augmentating(self, image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [self.IMG_SIZE, self.IMG_SIZE])
        image = tf.image.random_crop(image, size=[self.IMG_SIZE, self.IMG_SIZE, 3])
        image = tf.image.random_brightness(image, max_delta=0.5)
        return image, label
    
    def fit(self, X, y):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        X_train = list(X)
        y_train = list(y)

        for i in range(3):
            train_ds = dataset.map(self.augmentating, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            train_ds = train_ds.shuffle(1000).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

            for image, label in train_ds.as_numpy_iterator():
                X_train.append(image[0])
                y_train.append(label[0])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        return X_train, y_train



















