import time
import argparse
import os
import cv2
import numpy as np

from cv_utils import decode_yolov2, preprocess, draw_bounding_boxes
from tflite_runtime.interpreter import Interpreter
from flask import Flask, render_template, request, Response

app = Flask (__name__, static_url_path = '')

def process_age_gender(roi_img):

    ages = ['0-10', '11-20', '21-45', '46-60', '60-100']
    genders = ['M', 'F']

    results = second_stage_network.run(roi_img)
    age = np.argmax(results[0])
    gender = 0 if results[1] < 0.5 else 1

    label = f'{ages[age]} : {genders[gender]}'

    return label

class NetworkExecutor(object):

    def __init__(self, model_file):

        self.interpreter = Interpreter(model_file, num_threads=3)
        self.interpreter.allocate_tensors()
        _, self.input_height, self.input_width, _ = self.interpreter.get_input_details()[0]['shape']
        self.tensor_index = self.interpreter.get_input_details()[0]['index']

    def get_output_tensors(self):

      output_details = self.interpreter.get_output_details()
      tensor_indices = []
      tensor_list = []

      for output in output_details:
            tensor = np.squeeze(self.interpreter.get_tensor(output['index']))
            tensor_list.append(tensor)

      return tensor_list

    def run(self, image):
        if image.shape[1:2] != (self.input_height, self.input_width):
            img = cv2.resize(image, (self.input_width, self.input_height))
        img = preprocess(img)
        self.interpreter.set_tensor(self.tensor_index, img)
        self.interpreter.invoke()
        return self.get_output_tensors()

class Detector(NetworkExecutor):

    def __init__(self, label_file, model_file, threshold):
        super().__init__(model_file)
        self.threshold = float(threshold)

    def detect(self, frame):
        start_time = time.time()
        results = self.run(frame)
        elapsed_ms = (time.time() - start_time) * 1000

        detections = decode_yolov2(netout = results, nms_threshold = 0.1, threshold = self.threshold)
        draw_bounding_boxes(frame, detections, None, process_age_gender)

        fps  = 1 / elapsed_ms*1000
        print("Estimated frames per second : {0:.2f} Inference time: {1:.2f}".format(fps, elapsed_ms))

        return cv2.imencode('.jpg', frame)[1].tobytes()

@app.route("/")
def index():
   return render_template('index.html', name = None)

def gen(camera):
    while True:
        frame = camera.get_frame()
        image = detector.detect(frame)
        yield (b'--frame\r\n'+b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--first_stage', help='File path of .tflite file.', required=True)
parser.add_argument('--second_stage', help='File path of .tflite file.', required=True) 
parser.add_argument('--threshold', help='Confidence threshold.', default=0.5)
parser.add_argument('--source', help='picamera or cv', default='cv')
args = parser.parse_args()

if args.source == "cv":
    from camera_opencv import Camera
    source = 0
elif args.source == "picamera":
    from camera_pi import Camera
    source = 0
    
Camera.set_video_source(source)

detector = Detector(None, args.first_stage, args.threshold)
second_stage_network = NetworkExecutor(args.second_stage)

if __name__ == "__main__" :
   app.run(host = '0.0.0.0', port = 5000, debug = True)
    
