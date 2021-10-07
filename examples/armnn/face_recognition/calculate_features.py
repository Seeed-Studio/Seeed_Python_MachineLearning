# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Calculate face feature vector embedding and save it in a .json database under specified ID and name.

python3 run_video_file.py --fd_model_file_path YOLO_best_mAP.tflite --kp_model_file MobileFaceNet_kpts.tflite --video_file_path test_s.mp4 

"""

import os
import sys
import time
import json, base64
script_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(script_dir, '..', 'common'))

import cv2
import skimage 
import skimage.transform
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from yolov2 import yolo_processing, yolo_resize_factor
from utils import dict_labels
from cv_utils import init_video_file_capture, resize_with_aspect_ratio
from network_executor import ArmnnNetworkExecutor
import pyarmnn as ann

IMG_SHAPE = (128, 128) # in HW form 
offset_x = 0
offset_y = -15
src = np.array([(44+offset_x, 59+offset_y),
                (84+offset_x, 59+offset_y),
                (64+offset_x, 82+offset_y),
                (47+offset_x, 105),
                (81+offset_x, 105)], dtype=np.float32)

def write_db(db, id, name, vector):

    vector = base64.b64encode(vector).decode('utf-8')

    entry = {"name": name, "vector": vector}
    db[id] = entry
    print(db)
    f = open('database.db','w')
    entry = json.dumps(db)
    f.write(entry)
    f.close()

    return db

def read_db(db_path = 'database.db'):
    try:
        f = open(db_path, 'r')
    except FileNotFoundError:
        clear_db(db_path)
        f = open(db_path, 'r')
        
    content = f.read()
    if content:
        db = json.loads(content)
        for entry in db.keys():
            print(entry, ":", db[entry]['name'])
    f.close()
    return db

def clear_db(db_path = 'database.db'):

    f = open(db_path,'w')
    db = {}
    content = json.dumps(db)
    f.write(content)
    f.close()

def preprocess_input(x, **kwargs):
      x /= 127.5
      x -= 1.
      return x

def preprocess(frame: np.ndarray, input_binding_info: tuple):
    """
    Takes a frame, resizes, swaps channels and converts data type to match
    model input layer. The converted frame is wrapped in a const tensor
    and bound to the input tensor.

    Args:
        frame: Captured frame from video.
        input_binding_info:  Contains shape and data type of model input layer.

    Returns:
        Input tensor.
    """
    # Swap channels and resize frame to model resolution
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = resize_with_aspect_ratio(frame, input_binding_info)

    data_type = np.float32 if input_binding_info[1].GetDataType() == ann.DataType_Float32 else np.uint8
    resized_frame = np.expand_dims(np.asarray(resized_frame, dtype=data_type), axis=0)
    resized_frame = preprocess_input(resized_frame)
    assert resized_frame.shape == tuple(input_binding_info[1].GetShape())

    input_tensors = ann.make_input_tensors([input_binding_info], [resized_frame])
    return input_tensors

def process_faces(frame, detections, executor_kp, executor_fe, resize_factor, db, id, name):
    kpts_list = []

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for detection in detections:
        box = detection[1].copy()
        for i in range(len(box)):
            box[i] = int(box[i] * resize_factor)
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]

        face_img = frame[box[1]:box[3], box[0]:box[2]]

        face_img = cv2.resize(face_img, (128, 128)) 

        face_img = face_img.astype(np.float32)
        face_img = preprocess_input(face_img)

        input_tensors = ann.make_input_tensors([executor_kp.input_binding_info], [face_img])

        plist = executor_kp.run(input_tensors)[0][0]

        le = (x + int(plist[0] * w+5), y + int(plist[1] * h+5))
        re = (x + int(plist[2] * w), y + int(plist[3] * h+5))
        n = (x + int(plist[4] * w), y + int(plist[5] * h))
        lm = (x + int(plist[6] * w), y + int(plist[7] * h))
        rm = (x + int(plist[8] * w), y + int(plist[9] * h))
        kpts = [le, re, n, lm, rm]
        kpts_list.append(kpts)
        kpts = np.array(kpts, dtype = np.float32)

        transformer = skimage.transform.SimilarityTransform() 
        transformer.estimate(kpts, src) 
        M = transformer.params[0: 2, : ] 
        warped_img = cv2.warpAffine(frame, M, (IMG_SHAPE[1], IMG_SHAPE[0]), borderValue = 0.0) 

        face_img = warped_img.astype(np.float32)
        face_img = preprocess_input(face_img)

        input_tensors = ann.make_input_tensors([executor_fe.input_binding_info], [face_img])

        features = executor_fe.run(input_tensors)[0]

        write_db(db, id, name, features)

    return kpts_list

def draw_bounding_boxes(frame: np.ndarray, detections: list, resize_factor, kpts):
    """
    Draws bounding boxes around detected objects and adds a label and confidence score.

    Args:
        frame: The original captured frame from video source.
        detections: A list of detected objects in the form [class, [box positions], confidence].
        resize_factor: Resizing factor to scale box coordinates to output frame size.
        labels: Dictionary of labels and colors keyed on the classification index.
    """
    for detection in detections:
        class_idx, box, confidence = [d for d in detection]
        label, color = 'Person', (0, 255, 0)

        # Obtain frame size and resized bounding box positions
        frame_height, frame_width = frame.shape[:2]
        x_min, y_min, x_max, y_max = [int(position * resize_factor) for position in box]

        # Ensure box stays within the frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

        # Draw bounding box around detected object
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Create label for detected object class
        label = f'{label} {confidence * 100:.1f}%'
        label_color = (0, 0, 0) if sum(color)>200 else (255, 255, 255)

        # Make sure label always stays on-screen
        x_text, y_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0][:2]

        lbl_box_xy_min = (x_min, y_min if y_min<25 else y_min - y_text)
        lbl_box_xy_max = (x_min + int(0.55 * x_text), y_min + y_text if y_min<25 else y_min)
        lbl_text_pos = (x_min + 5, y_min + 16 if y_min<25 else y_min - 5)

        # Add label and confidence value
        cv2.rectangle(frame, lbl_box_xy_min, lbl_box_xy_max, color, -1)
        cv2.putText(frame, label, lbl_text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.50,
                    label_color, 1, cv2.LINE_AA)

        for kpt_set in kpts:
            for kpt in kpt_set:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 5, (255, 0, 0), 2)


def main(args):

    #clear_db()
    db = read_db(args.db_file_path)

    executor_fd = ArmnnNetworkExecutor(args.fd_model_file_path, args.preferred_backends)
    executor_kp = ArmnnNetworkExecutor(args.kp_model_file_path, args.preferred_backends)    
    executor_fe = ArmnnNetworkExecutor(args.fe_model_file_path, args.preferred_backends) 

    process_output = yolo_processing

    image = cv2.imread(args.picture_file_path)

    frame_height, frame_width = image.shape[:2]
    model_height, model_width = list(executor_fd.input_binding_info[1].GetShape())[1:3]
    resize_factor = max(frame_height, frame_width) / max(model_height, model_width)

    input_tensors = preprocess(image, executor_fd.input_binding_info)

    output_result = executor_fd.run(input_tensors)
    detections = process_output(output_result)
    kpts = process_faces(image, detections, executor_kp, executor_fe, resize_factor, db, args.id, args.name)

    draw_bounding_boxes(image, detections, resize_factor, kpts)

    cv2.imwrite(args.picture_file_path.split('.')[0]+'_result.jpg', image)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--picture_file_path', required=True, type=str,
                        help='Path to the picture file to calculate feature embedding from')
    parser.add_argument('--db_file_path', type=str,
                        help='Path to the database file with feature embedding vectors')

    parser.add_argument('--id', default = '0', type=str,
                        help='Unique ID for the face')
    parser.add_argument('--name', default = 'John Doe', type=str,
                        help='Name for the face feature vector (can be duplicate)')

    parser.add_argument('--fd_model_file_path', required=True, type=str,
                        help='Path to the YOLOv2 face detection model to use')
    parser.add_argument('--kp_model_file_path', required=True, type=str,
                        help='Path to the keypoints detection model to use')
    parser.add_argument('--fe_model_file_path', required=True, type=str,
                        help='Path to the feature vector embedding extractor model to use')

    parser.add_argument('--preferred_backends', type=str, nargs='+', default=['CpuAcc', 'CpuRef'],
                        help='Takes the preferred backends in preference order, separated by whitespace, '
                             'for example: CpuAcc GpuAcc CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc]. '
                             'Defaults to [CpuAcc, CpuRef]')
    args = parser.parse_args()
    main(args)
