from flask import Flask, Response, render_template_string
import mediapipe as mp
import SquatPosture as sp
import cv2
import tensorflow as tf
import numpy as np
from utils import landmarks_list_to_array, label_params, label_final_results

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

model = tf.keras.models.load_model("working_model.keras")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

def gen(camera):
    cap = camera.video
    i=0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)

            if not success:
                print("Ignoring empty camera frame.")
                break

            image_height, image_width, _ = image.shape

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            dim=(image_width//5, image_height//5)

            resized_image = cv2.resize(image, dim)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            results = pose.process(resized_image)

            params = sp.get_params(results)
            flat_params = np.reshape(params, (5, 1))

            output = model.predict(flat_params.T)

            output[0][0] *= 0.7
            output[0][1] *= 1.7
            output[0][2] *= 4
            output[0][3] *= 0
            output[0][4] *= 5

            output = output * (1 / np.sum(output))

            output_name = ['c', 'k', 'h', 'r', 'x', 'i']

            output[0][2] += 0.1

            label = ""

            for i in range(1, 4):
                label += output_name[i] if output[0][i] > 0.5 else ""

            if label == "":
                label = "c"

            label += 'x' if output[0][4] > 0.15 and label=='c' else ''

            label_final_results(image, label)

            i+=1

            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FITFORM</title>
    <link rel="stylesheet" href="/static/stylesheet.css">
</head>
<body>
    <div class="main">
        <div class="main-container">
            <table cellspacing="20px" class="table">
                <tr class="row">
                    <td> <img src="/static/logo.png" class="logo" /> </td>
                </tr>
                <tr class="choices">
                    <td> Your personal AI Gym Trainer </td>
                </tr>
                <tr class="row">
                    <td> <img src="/video_feed" class="feed"/> </td>
                </tr>
                <tr class="disclaimer">
                    <td> Please ensure that the scene is well lit and your entire body is visible </td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
