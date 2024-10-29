from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import io
from PIL import Image
import cvlib as cv
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load your trained model
model = load_model('gender_detection.h5')
classes = ['man', 'woman']

# Video capture for real-time streaming
video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces in the image
        faces, confidences = cv.detect_face(frame)

        for face in faces:
            (startX, startY, endX, endY) = face

            # Crop the face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            # Preprocess the face image for the model
            face_crop = cv2.resize(face_crop, (50, 50))
            face_crop = face_crop.astype("float32") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Predict gender
            prediction = model.predict(face_crop)[0]
            label_idx = np.argmax(prediction)
            label = classes[label_idx]

            # Draw rectangle around the face and put the label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream the video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
