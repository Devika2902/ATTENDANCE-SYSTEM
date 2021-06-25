import face_recognition
import cv2
import numpy as np

print("loading known faces")

img1=face_recognition.load_image_file("known_faces/Virat Kohli.jpg")
im1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file("unknown_faces/unknown1.jpg")
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(img1)[0]
encode1=face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encode1],encodeTest)
faceDis=face_recognition.face_distance([encode1],encodeTest)
#lower the distance- better match
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow("known1",img1)
cv2.imshow("unknown1",imgTest)
cv2.waitKey(0)








