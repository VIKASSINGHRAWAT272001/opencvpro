import cv2
import numpy as np

# Load Haarcascade face classifier
# Github Repo for various other haarcascade classifiers - https://github.com/opencv/opencv/tree/master/data/haarcascades

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_crop(img):
    
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces == ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

cam = cv2.VideoCapture(0)   #Starting Webcam
count = 0


# Collecting 200 samples of your face from webcam input
while True:

    ret, frame = cam.read()
    if face_crop(frame) is not None:
        count += 1
        face = cv2.resize(face_crop(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Saving the samples in a directory
        file_name_path = 'DALE/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # display live count for number of samples
        cv2.putText(face, str(count), (25, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        cv2.imshow('Cropped Face b/w ', face)

    if cv2.waitKey(1) == 13 or count == 200:  # Close either when ENTER key pressed or number of samples reach 200.
        break

cam.release()
cv2.destroyAllWindows()      
print(" Face Samples Collected Successfully")


import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Training dataset Location
data_path = 'DALE/'

data_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []   # Empty lists or arrays for storing training data and labels

# Creating a numpy array for training data
for i, files in enumerate(data_files):
    image_path = data_path + data_files[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

#Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face_LBPHFaceRecognizer.create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")

import cv2
import numpy as np
import os
import time
import pywhatkit
import smtplib
import threading

def linst():
    # Creating an ec2 instance on aws cloud
    os.system("aws ec2 run-instances  --image-id ami-0ad704c126371a549 --instance-type t2.micro  --subnet-id subnet-183f4154  --count 1 --security-group-ids sg-02d89cdfa8237d495  --key-name d  > ec2.txt")
    print("Instance Launched")
            
    # Creating volume (5gb)
    os.system("aws ec2 create-volume --availability-zone ap-south-1 --size 5 --volume-type gp2 --tag-specification  ResourceType=volume,Tags=[{Key=face,Value=volume}]  > ebs.txt")
    print("Volume Created of size 5 gb")
    print("Initiating in 120 seconds")
    time.sleep(120)
    ec2_id = open("ec2.txt", 'r').read().split(',')[3].split(':')[1].split('"')[1]
    ebs_id = open("ebs.txt", 'r').read().split(',')[6].split(':')[1].split('"')[1]
            
    os.system("aws ec2 attach-volume --instance-id   " + ec2_id +"  --volume-id  " + ebs_id  +"  --device /dev/sdf")
    print("Volume Successfully attached to the instance")
    
    
def smes():
    # Sending Whatsapp Message
    pywhatkit.sendwhatmsg_instantly(phone_no="+91**********", 
                    message="Hi, arth learner")

    # Sending email
    print("Whatsapp Message sent Successfully!!")
    pywhatkit.send_mail(email_sender= "emailid",
                    password= "**********",
                    subject="Automated E-mail",
                    message="hello everyone",
                    email_receiver="receiver id")
    

launch_instance_t1 = threading.Thread(target=linst)  
send_message_t1 = threading.Thread(target=smes)


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces == ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

# Open Webcam
cap = cv2.VideoCapture(0)
while True:
    
    launch_instance = False
    send_messages = False

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        
        # "results" comprises of a tuple containing the label and the confidence value
        results = model.predict(face)
        
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        
        
        if confidence > 89:
            cv2.destroyAllWindows()
            send_messages = True
            send_message_t1.start()
            cv2.putText(image, display_string, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,0), 2)
            cv2.putText(image, "WELLCOME VIKAS RAWAT", (200, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            cv2.waitKey(20000)
            cv2.destroyAllWindows()
            break
         
        else:
            print("Second face detected - Initiating aws ec2 instance...")
            cv2.destroyAllWindows()
            launch_instance = True
            launch_instance_t1.start()
            image2 = image
            cv2.putText(image2, "Second face detected", (50, 50) , cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255), 2)
            cv2.putText(image2, "Initiating aws ec2 instance...", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255), 2)
            cv2.imshow('Second Face Detected', image2 )
            cv2.waitKey(20000)
            cv2.destroyAllWindows()
            break
            
        if launch_instance == True or send_message == True:
            break

    except:
        cv2.putText(image, "Face Detection Failed", (50, 50) , cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Looking For Face", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        
        pass
        
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
