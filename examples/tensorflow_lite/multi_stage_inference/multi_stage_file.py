import time
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from cv_utils import init_video_file_capture, decode_yolov2, decode_classifier, draw_classification, draw_bounding_boxes, preprocess
from tflite_runtime.interpreter import Interpreter

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

def main(args):
    video, video_writer, frame_count = init_video_file_capture(args.file, 'age_gender_demo')

    frame_num = len(frame_count)
    times = []

    for _ in tqdm(frame_count, desc='Processing frames'):
        frame_present, frame = video.read()
        if not frame_present:
            continue

        start_time = time.time()
        
        results = first_stage_network.run(frame)
        detections = decode_yolov2(netout = results, nms_threshold = 0.1, threshold = args.threshold)
        draw_bounding_boxes(frame, detections, None, process_age_gender)

        elapsed_ms = (time.time() - start_time) * 1000

        times.append(elapsed_ms)
        video_writer.write(frame)

    print('Finished processing frames')
    video.release(), video_writer.release()

    print("Average time(ms): ", sum(times)//frame_num) 
    print("FPS: ", 1000.0 / (sum(times)//frame_num)) # FPS = 1 / time to process loop

if __name__ == "__main__" :

    print("OpenCV version: {}".format(cv2. __version__))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--first_stage', help='File path of .tflite file.', required=True)
    parser.add_argument('--second_stage', help='File path of .tflite file.', required=True)    
    parser.add_argument('--threshold', help='Confidence threshold.', default=0.7)
    parser.add_argument('--file', help='File path of video file', required=True)
    args = parser.parse_args()

    first_stage_network = NetworkExecutor(args.first_stage)
    second_stage_network = NetworkExecutor(args.second_stage)

    main(args)
    
