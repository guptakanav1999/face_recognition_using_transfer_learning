from keras.models import load_model
model = load_model('facedetect.h5')
import cv2
import numpy as np
#import Ipython.display as ipd
import PIL
from keras.preprocessing import image
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_detector(img,size=0.5):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(224,224))
    return img,roi
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)
    face = np.array(face)
    face = np.expand_dims(face,axis=0)
    if(face.shape == (1,0)):
        cv2.putText(image, "i dont know", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
    else:
        results = model.predict(face)
        print(results)
        if(results[0][0] == 1.0):
            cv2.putText(image, "diksha", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
        if(results[0][0] != 1.0):
            cv2.putText(image, "kanav", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()