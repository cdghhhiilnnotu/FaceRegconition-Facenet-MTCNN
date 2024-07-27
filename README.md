# FACE RECOGNITION REPOSITORY - MODULE RECOGNITION WITH FACENET + MTCNN

- Create dataset path: This section explains how to structure your dataset folder.
  + The main folder is named /dataset.
  + Inside /dataset, there are subfolders named class1, class2, etc. (presumably representing different people's images).
  + Each class folder contains image files named img.* (where * can be any number or extension, like .jpg).

- More likes:<br>
----/dataset\n<br>
--------/class1<br>
------------img.*<br>
--------/class2<br>
------------img.*<br>
----main.py<br>
----lib.py<br>
    
- Change './Hau-Face' in main.py to dataset name:
  + This instructs you to modify the code in the file main.py.
  + You need to replace the text './Hau-Face' with the actual name of your dataset folder
  + (e.g., if your main folder is named People, replace './Hau-Face' with './People').
 
- Install packages from requirements.txt: pip install -r requirements.txt
