from lib import *
from modules import *

faceloader= FaceLoader('./Hau-Face')
X, y = faceloader.load_classes()

facereg = FaceReg()

facereg.training(X, y)

random_name = os.listdir("./Hau-Face")[random.randint(0, len(os.listdir("./Hau-Face")) - 1)]
random_file = os.listdir(f"./Hau-Face/{random_name}")[random.randint(0, len(os.listdir(f"./Hau-Face/{random_name}")) - 1)]
t_im = cv.imread(f"./Hau-Face/{random_name}/{random_file}")
print(f"./Hau-Face-Test/{random_name}/{random_file}")

print(facereg.predict(t_im))










