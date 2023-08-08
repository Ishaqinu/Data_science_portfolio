import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from twilio.rest import Client
import pandas as pd
import json

details_of_students = pd.read_csv("/home/ishaq/DL/attendence project/details.csv")

with open('credentials.json') as file:
    credentials = json.load(file)

account_sid = credentials['account_sid']
auth_token = credentials['auth_token']
twilio_phone_number= credentials['twilio_phone_number']
client = Client(account_sid, auth_token)

path = '/home/ishaq/DL/attendence project/training_images'
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(path + '/' + cl)
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)

def findencodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('attendance.csv', 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{dtString}\n')


def getAbsentStudentNames(classnames):
    absent_students = []
    with open('attendance.csv', 'r') as f:
        attendance_data = f.readlines()
        attendance_names = [line.split(',')[0] for line in attendance_data]

    for class_name in classnames:
        if class_name not in attendance_names:
            absent_students.append(class_name)
    return absent_students



def sendAbsentMessage(absent_students):
    for student in absent_students:
        student_details = details_of_students[details_of_students['name'] == student]
        phone_number = student_details['phone'].values[0]
        message = client.messages.create(
            body=f"You were absent in today's class. Please contact your instructor for further information.",
            from_=twilio_phone_number,
            to=phone_number
        )
        print(f"Sent message to {phone_number}: {message.sid}")


encodeListKnown = findencodings(images)
print('Encoding Complete')

video = cv2.VideoCapture(0)

while True:
    success, img = video.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    absent_students = getAbsentStudentNames(classnames)
    cv2.imshow('Webcam', img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

print(absent_students)
sendAbsentMessage(absent_students)
