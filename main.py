import numpy as np
import cv2
import os
import sqlite3

facedetect = cv2.CascadeClassifier("C:/Users/lenovo/Desktop/Project_Attendence/Face Detection datasetCreator/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/Users/lenovo/Desktop/Project_Attendence/Face recognition Trainer/training_model.yml")

#creting function to get profile from sql database named students
def getprofile(id):  
    conn = sqlite3.connect("students.db")
    cursor = conn.execute("SELECT * FROM students WHERE id=?",(id,))  # This line performs a SQL query to retrieve the record(s) from the students table where the id matches the provided id

    profile = None  #keeping profile none for now
    for row in cursor:
        profile = row    #The result row is a tuple representing the entire row (with id, Name, Age, Gender).
    conn.close()    
    return profile 

while(True):
    ret,img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
  
  
    # Flip the image horizontally (mirror effect)
    img = cv2.flip(img, 1)


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,3)

    for (x, y, w, h) in faces:
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), 2)
        id,conf = recognizer.predict(gray[y:y+h , x:x+w])    #see about this line at end of this code
        if(conf>75):
            continue
        profile = getprofile(id)
        print(profile)
        cv2.putText(img, f"Name: {str(profile[2])}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"Age: {str(profile[1])}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1, cv2.LINE_AA)


    cv2.imshow("Live Camera",img)
    if(cv2.waitKey(1)==ord("q")):
        break

cam.release()
cv2.destroyAllWindows()











# recognizer.predict():
#    - The `recognizer` is an object of a face recognizer, typically created by `cv2.face.LBPHFaceRecognizer_create()` or any other recognizer type.
#    - The `predict()` method is used to **predict** the identity of a face (based on the training data) that has been detected in the image.

# Arguments for `predict()`:
#    - `gray[y:y+h, x:x+w]` is the region of the input image (`gray`) that contains the face (extracted from the bounding box coordinates `(x, y)` for the top-left corner and `(x+w, y+h)` for the bottom-right corner).
#      - `gray` is the grayscale image.
#      - `y:y+h` and `x:x+w` are the coordinates of the face region within the image, where `(x, y)` is the top-left corner and `(w, h)` represent the width and height of the bounding box.
#    - `gray[y:y+h, x:x+w]` slices the image to extract the region containing the face.
#    - The `predict()` method uses this **face region** to make a prediction (i.e., recognize the face).

# 3. **Return Values**:
#    - The `predict()` method returns two values:
#      - **`id`**: The predicted ID of the recognized person from the training data. This ID corresponds to the label you assigned to the faces during the training process (e.g., `User.1.1.jpg` would have ID `1`).
#      - **`conf`**: The confidence level of the prediction, which represents how certain the recognizer is about the prediction. The lower the `conf`, the more confident the model is in its prediction. Typically, a threshold is set for `conf` to decide whether the prediction is reliable or not.

# ### Example of how `predict()` works:
# If the `predict()` method returns:
# - `id = 1`, and
# - `conf = 45.6`,

# It means that the face detected corresponds to the person with ID `1` in the training data, and the recognizer is 45.6% confident in its prediction. You can set a threshold for `conf` (like `conf < 50`), below which the recognition is considered unreliable.

  


