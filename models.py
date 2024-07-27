from lib import *

class FaceLoader:

    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        faces = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                faces.append(single_face)
            except:
                pass
        return faces
    
    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory + '/' + sub_dir + '/'
            faces = self.load_faces(path)
            labels = [sub_dir for _ in range(len(faces))]
            print(f"Loaded successfully: {len(faces)}")
            self.X.extend(faces)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)


class FaceReg():
    
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
        embbedded_X = []
        for img in X:
            embbedded_X.append(self.embedding(img))
        self.X = np.asarray(embbedded_X)

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

        emb_img = [emb_img]
        y_preds = self.model.predict(emb_img)
        y_preds = self.encoder.inverse_transform(y_preds)
        return y_preds

    def accuracy(self):
        ypreds = self.model.predict(self.X)
        return accuracy_score(self.y, ypreds)
        



