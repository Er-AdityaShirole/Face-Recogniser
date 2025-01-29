
import cv2  # Import OpenCV library for computer vision tasks

import sqlite3  # Import SQLite3 library for database operations

# Load the Haar Cascade Classifier for face detection
faceDetect =cv2.CascadeClassifier('C:\\Users\\lenovo\\Desktop\\Project_Attendence\\Face Detection datasetCreator\\haarcascade_frontalface_default.xml')  #cascadclassifier() is function who initializes a Haar Cascade Classifier.It takes the path to an XML file containing the pre-trained model data.


# Initialize the webcam for capturing video
cam = cv2.VideoCapture(0)

# Function to insert or update user details in the database
def insertOrUpdate(Id, Name, Age, Gen):
    # Connect to SQLite3 database. If it doesn't exist, it will be created
    conn = sqlite3.connect("students.db")
    
    # Query to check if a record with the given Id already exists
    cmd = "SELECT * FROM students WHERE ID=" + str(Id)
    cursor = conn.execute(cmd)  # Execute the query
    
    # Check if a record exists for the given Id
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1  # If a record is found, set flag to 1
    
    # If record exists, update the details
    if isRecordExist == 1:
        conn.execute("UPDATE students SET Name=? WHERE id=?", (Name, Id,))
        conn.execute("UPDATE students SET Age=? WHERE id=?", (Age, Id,))
        conn.execute("UPDATE students SET Gender=? WHERE id=?", (Gen, Id,))
    else:
        # If no record exists, insert a new record into the database
        conn.execute("INSERT INTO students(id, Name, Age, Gender) VALUES (?, ?, ?, ?)", (Id, Name, Age, Gen))
    
    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

# Collect user information from input
Id = input('Enter User Id: ')  # User's unique ID
name = input('Enter User Name: ')  # User's name
age = input('Enter User Age: ')  # User's age
gen = input('Enter User Gender: ')  # User's gender

# Call the insertOrUpdate function to add/update the user's information in the database
insertOrUpdate(Id, name, age, gen)

# Variable to count the number of samples (face images) collected
sampleNum = 0
frame_count =0#for skipping frames

# Start capturing video and processing frames
while True:
   
    # Read a frame from the webcam
    ret, img = cam.read()
    #can see about ret on gpt, it is boolean value which tell if frame in img is captured or failed
    if not ret:
        print("Failed to grab frame")
        break  # Exit the loop if frame capture fails
    
    frame_count += 1
    
    # Skip every other frame
    if frame_count % 2 != 0:
        continue    

    # Resize the image to a smaller resolution (e.g., 640x480) for BETTER OUTPUT
    img = cv2.resize(img, (640, 480))
    
    # Convert the frame to grayscale (necessary for Haar Cascade detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
   
   
    # Detect faces in the frame
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)  # Scale factor=1.1, minNeighbors=4  can see about these two here:https://chatgpt.com/share/67900776-e3c0-8000-b794-c5fa3075f8dd
#     What is faces?
# The variable faces stores the bounding boxes of detected faces in the image.
# Each detected face is represented as a tuple (x, y, w, h):
# x, y: Top-left corner of the detected face.
# w, h: Width and height of the bounding box around the detected face.
# If multiple faces are detected, faces will contain multiple tuples, one for each detected face.
    
    
    
    # Loop through all detected faces
    for (x, y, w, h) in faces:
        sampleNum += 1  # Increment sample number
        # Save the detected face region as an image in the "dataSet" directory
        cv2.imwrite("dataSet/User." + str(Id) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w]) #this format in which image will be saved: dataSet/User.<Id>.<sampleNum>.jpg
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), 2)   #(255,255,255) represents color code of white & 2 is thickness of border
        #cv2.waitKey(400)  # Wait for 400 milliseconds before processing the next face
    
    # Display the video feed with rectangles around detected faces
    cv2.imshow("Face", img)
    cv2.waitKey(1)  # Short delay for key events
    
    # Stop if 120 samples have been collected
    if sampleNum > 120:
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()




#This code detetcs faces in while loop and 
# in for loop inside the while loop: jpg images are saved in dataSet named folder in a very specific format which
# also includes id and sample number
# while loop runs till it collects 120 sample images  