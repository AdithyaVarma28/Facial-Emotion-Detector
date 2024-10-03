import cv2
import numpy as np
import os

def flattening(face_image):
    return face_image.flatten()

def image_matrix(directory, target_size):
    matrix=[]
    labels=[]
    for emotion in os.listdir(directory):
        emotion_directory=os.path.join(directory,emotion)
        for image in os.listdir(emotion_directory):
            if image.lower().endswith('.jpg'):
                image_path=os.path.join(emotion_directory,image)
                img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE) 
                img_resized=cv2.resize(img,target_size)
                matrix.append(flattening(img_resized))
                labels.append(emotion) 
    return np.array(matrix),np.array(labels)

def training(target_size):
    training_matrix,labels=image_matrix('Dataset/train',target_size)
    mean=np.mean(training_matrix,axis=0)
    centered_matrix=training_matrix-mean
    U,V,Vector=np.linalg.svd(centered_matrix,full_matrices=False)
    projected_matrix=np.dot(centered_matrix,Vector.T)
    return mean,Vector,projected_matrix,labels

def recognize_face(validation_matrix,mean,Vector,training_projected_matrix):
    centered_matrix=validation_matrix-mean
    validation_projected_matrix=np.dot(centered_matrix,Vector.T)
    euclidean_distances=np.linalg.norm(training_projected_matrix-validation_projected_matrix,axis=1)
    return euclidean_distances

def recognize_from_webcam(target_size=(48,48)):
    mean,Vector,training_projected_matrix,labels=training(target_size)
    cam=cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    while True:
        ret,frame=cam.read()
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5)
        for (x,y,w,h) in faces:
            face=gray_frame[y:y+h,x:x+w] 
            face_resized=cv2.resize(face,target_size)
            face_flat=flattening(face_resized)
            euclidean_distances=recognize_face(face_flat,mean,Vector,training_projected_matrix)
            nearest_index=np.argmin(euclidean_distances)
            recognized_emotion=labels[nearest_index]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.putText(frame,recognized_emotion,(x,y-10),cv2.FONT_HERSHEY_DUPLEX,0.6,(0,255,0),1)
        cv2.imshow('Face Camera',frame)
        if cv2.waitKey(1)==ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    recognize_from_webcam(target_size=(48,48))
