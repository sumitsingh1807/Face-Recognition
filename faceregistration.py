import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np
import sqlite3

# Initialize MTCNN face detector and FaceNet embedder
detector = MTCNN()
embedder = FaceNet()

def capture_face_and_get_embedding():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(rgb_frame)
        if len(faces) > 0:
            x, y, width, height = faces[0]['box']
            face = rgb_frame[y:y+height, x:x+width]
            face = cv2.resize(face, (160, 160))
            embedding = embedder.embeddings([face])
            return embedding[0]

        cv2.imshow('Face Registration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def store_user_embedding(username, embedding):
    conn = sqlite3.connect('smart_shopping.db')
    cursor = conn.cursor()

    cursor.execute("INSERT INTO users (username, face_embedding) VALUES (?, ?)", 
                   (username, embedding.tobytes()))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    username = input("Enter the username: ")
    print("Capturing face... Please look at the camera.")
    embedding = capture_face_and_get_embedding()

    if embedding is not None:
        store_user_embedding(username, embedding)
        print(f"User {username} registered successfully.")
    else:
        print("No face detected.")