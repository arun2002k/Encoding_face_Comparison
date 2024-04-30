import cv2
import face_recognition
import pickle

def load_face_encodings(file):
    with open(file, 'rb') as f:
        face_encodings = pickle.load(f)
    return face_encodings

def main():
    video_capture = cv2.VideoCapture(0)  # 0 represents the default camera
    known_face_encodings = load_face_encodings("face_encodings.pkl")

    while True:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            results = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in results:
                print("Yes")  # Person recognized
            else:
                print("No")   # Person not recognized

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
