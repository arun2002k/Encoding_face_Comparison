import cv2
import face_recognition
import pickle

def encode_and_store_faces(output_file):
    known_face_encodings = []

    video_capture = cv2.VideoCapture(0)  # 0 represents the default camera

    while True:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            known_face_encodings.append(face_encoding)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    with open(output_file, 'wb') as file:
        pickle.dump(known_face_encodings, file)

if __name__ == "__main__":
    output_file = "face_encodings.pkl"
    encode_and_store_faces(output_file)
