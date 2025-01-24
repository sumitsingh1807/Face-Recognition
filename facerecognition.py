import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Suppress TensorFlow informational logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2

from mtcnn import MTCNN
from keras_facenet import FaceNet
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize MTCNN face detector and FaceNet embedder
detector = MTCNN()
embedder = FaceNet()

def get_face_embedding(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    if len(faces) > 0:
        x, y, width, height = faces[0]['box']
        face = rgb_frame[y:y+height, x:x+width]
        face = cv2.resize(face, (160, 160))
        embedding = embedder.embeddings([face])
        return embedding[0]
    return None

def get_all_users():
    conn = sqlite3.connect('smart_shopping.db')
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, username, face_embedding FROM users")
    users = cursor.fetchall()

    user_data = []
    for user in users:
        user_id = user[0]
        username = user[1]
        embedding = np.frombuffer(user[2], dtype=np.float32)
        user_data.append((user_id, username, embedding))

    conn.close()
    return user_data

def recognize_user(embedding, threshold=0.6):
    users = get_all_users()

    for user_id, username, stored_embedding in users:
        similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
        if similarity > threshold:
            return user_id, username

    return None, None

if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        embedding = get_face_embedding(frame)

        if embedding is not None:
            user_id, username = recognize_user(embedding)
            if username:
                print(f"Recognized User: {username}")
                break
            else:
                print("User not recognized.")
                break

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
