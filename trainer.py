import os
import numpy as np
import cv2
from PIL import Image 

recognizer = cv2.face.LBPHFaceRecognizer_create()     #-> READ THIS IN info_recognizer.py
path =  "dataSet"


def get_images_id(path):
    images_paths = [os.path.join(path,f) for f in os.listdir(path)]   #.................---->        os.listdir(path) :  Returns a list of all the files and directories in the specified directory (path).                                                        
    faces = []                                                                                               # os.path.join(path, f) : Joins the path (the directory name) with each file name (f) to create a full file path. ex. if path(dataSet) directory have file called 1.jpg then this function returns : dataSet\\1.jpg
    ids =[]
    for each_img_path in images_paths:
        faceimg = Image.open(each_img_path).convert("L")     #image converted to gray color and stored in faceimg
        facenp = np.array(faceimg,np.uint8)
        id = os.path.split(each_img_path)[-1].split(".")[1]                                                                                           
        print(id)
        faces.append(facenp)
        ids.append(int(id))
        cv2.imshow("Training",facenp)
        cv2.waitKey(10)

    return faces, np.array(ids)    


faces,ids = get_images_id(path)
recognizer.train(faces,ids)       #given a list of faces and numpyArray of ids .....in list of faces each element is numpyArray of image
recognizer.save("C:/Users/lenovo/Desktop/Project_Attendence/Face recognition Trainer/training_model.yml") #saving trained model

cv2.destroyAllWindows
