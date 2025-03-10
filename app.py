import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from flask import Flask, Response, jsonify, render_template
from time import sleep
import os

current_directory = os.getcwd()
model_path = os.path.join(current_directory, "mobilenetv2-epoch_10.hdf5")
predict_path = os.path.join(current_directory, "shape_predictor_68_face_landmarks.dat")

app = Flask(__name__)

model = load_model(model_path)
predictor = dlib.shape_predictor(predict_path)

EAR_THRESHOLD = 0.2
EYE_AR_CONSEC_FRAMES = 2


current_status = "No Real Human Detected"


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def adjust_brightness(frame, alpha=1.2, beta=0):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def predict_liveness(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = img_to_array(face_img)
    face_img = face_img.astype("float32") / 255.0
    face_img = face_img.reshape((1, 224, 224, 3))
    prediction = model.predict(face_img)
    return prediction[0][0] > 0.5


def generate_frames():
    cap = cv2.VideoCapture(0)
    blink_counter = 0
    detector = dlib.get_frontal_face_detector()

    global current_status

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = adjust_brightness(frame, alpha=1.2, beta=0)
        faces = detector(frame)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_img = frame[y : y + h, x : x + w]

            if face_img.size == 0:
                continue

            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            landmarks = predictor(gray_face, dlib.rectangle(0, 0, w, h))

            left_eye = [
                (landmarks.part(i).x - x, landmarks.part(i).y - y)
                for i in range(36, 42)
            ]
            right_eye = [
                (landmarks.part(i).x - x, landmarks.part(i).y - y)
                for i in range(42, 48)
            ]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    print("Blink detected")
                blink_counter = 0

            is_real = predict_liveness(face_img)

            if is_real and blink_counter >= EYE_AR_CONSEC_FRAMES:
                current_status = "Real Human Detected"
                print("Real human detected")
                sleep(10)
            else:
                current_status = "No Real Human Detected"

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")

    cap.release()


@app.route("/status")
def get_status():
    """API endpoint to get the current status"""
    return jsonify({"status": current_status})


@app.route("/")
def index():
    """Main route that serves the video stream"""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Route to serve video feed"""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
