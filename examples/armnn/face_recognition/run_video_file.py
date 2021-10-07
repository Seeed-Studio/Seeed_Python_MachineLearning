# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# Modified by Seeed Studio 2021 Dmitry Maslov
# SPDX-License-Identifier: MIT

"""
Face recognition demo that takes a video file, runs inference on each frame producing
bounding boxes and ID numbers corresponding to entries in database, and saves the processed video.

python3 run_video_file.py --video_file_path test_s.mp4 --db_file_path database.db --fd_model_file_path ../face_rec_models/YOLOv2_best_mAP.tflite --kp_model_file_path ../face_rec_models/MobileFaceNet_kpts.tflite --fe_model_file_path ../face_rec_models/MobileFaceNet_features.tflite
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
from cv_utils import init_video_file_capture, resize_with_aspect_ratio, preprocess
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

def preprocess_input(x, **kwargs):
      x /= 127.5
      x -= 1.
      return x

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

def process_faces(frame, detections, executor_kp, executor_fe, resize_factor, db):
    kpts_list = []
    id_list = []

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for detection in detections:
        box = detection[1].copy()
        for i in range(len(box)):
            box[i] = int(box[i] * resize_factor)

        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]

        frame_height, frame_width = frame.shape[:2]
        x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]

        # Ensure box stays within the frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

        face_img = frame[y_min:y_max, x_min:x_max]

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
        highest_score = 0

        for id in db.keys():
            cos_sim = np.dot(features, db[id]['vector'])/(np.linalg.norm(features)*np.linalg.norm(db[id]['vector']))
            cos_sim /= 2
            cos_sim += 0.5
            cos_sim *= 100
            if highest_score < cos_sim:
                highest_score = cos_sim
                recognized_id = id
                
        if highest_score > 80.0:
            #print(recognized_id, db[recognized_id]['name'], highest_score)
            id_list.append([recognized_id, db[recognized_id]['name'], highest_score])
        else:
            id_list.append(['X', '', 0.0])
    return kpts_list, id_list

def draw_bounding_boxes(frame: np.ndarray, detections: list, resize_factor, kpts, ids):

    for i in range(len(detections)):
        class_idx, box, confidence = [d for d in detections[i]]
        color = (0, 125, 0) if ids[i][2] > 0 else (0, 0, 255)

        # Obtain frame size and resized bounding box positions
        frame_height, frame_width = frame.shape[:2]
        x_min, y_min, x_max, y_max = [int(position * resize_factor) for position in box]

        # Ensure box stays within the frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

        # Draw bounding box around detected object
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Create label for detected object class
        label = 'ID: {} Name: {} {}%'.format(*ids[i])
        label_color = (255, 255, 255)

        # Make sure label always stays on-screen
        x_text, y_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0][:2]

        lbl_box_xy_min = (x_min, y_min if y_min<25 else y_min - y_text)
        lbl_box_xy_max = (x_min + int(0.75 * x_text), y_min + y_text if y_min<25 else y_min)
        lbl_text_pos = (x_min + 5, y_min + 16 if y_min<25 else y_min - 5)

        # Add label and confidence value
        cv2.rectangle(frame, lbl_box_xy_min, lbl_box_xy_max, color, -1)
        cv2.putText(frame, label, lbl_text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.70, label_color, 1, cv2.LINE_AA)

        for kpt_set in kpts:
            for kpt in kpt_set:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 5, (255, 0, 0), 2)

def main(args):
    video, video_writer, frame_count = init_video_file_capture(args.video_file_path, 'face_recognition_demo')
    frame_num = len(frame_count)

    db = read_db(db_path = 'database.db')
    for item in db:
        db[item]['vector'] = np.frombuffer(base64.b64decode(db[item]['vector']), np.float32)

    executor_fd = ArmnnNetworkExecutor(args.fd_model_file_path, args.preferred_backends)
    executor_kp = ArmnnNetworkExecutor(args.kp_model_file_path, args.preferred_backends)    
    executor_fe = ArmnnNetworkExecutor(args.fe_model_file_path, args.preferred_backends) 

    process_output, resize_factor = yolo_processing, yolo_resize_factor(video, executor_fd.input_binding_info)

    times = []

    for _ in tqdm(frame_count, desc='Processing frames'):
        frame_present, frame = video.read()
        if not frame_present:
            continue

        input_tensors = preprocess(frame, executor_fd.input_binding_info)

        start_time = time.time()
        output_result = executor_fd.run(input_tensors)
        detections = process_output(output_result)
        kpts, ids = process_faces(frame, detections, executor_kp, executor_fe, resize_factor, db)
        end_time = (time.time() - start_time)*1000
        times.append(end_time)

        draw_bounding_boxes(frame, detections, resize_factor, kpts, ids)
        video_writer.write(frame)

    print('Finished processing frames')
    video.release(), video_writer.release()

    print("Average time(ms): ", sum(times)//frame_num) 
    print("FPS: ", 1000.0 / (sum(times)//frame_num)) # FPS = 1 / time to process loop

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_file_path', required=True, type=str,
                        help='Path to the video file to run object detection on')
    parser.add_argument('--db_file_path', type=str,
                        help='Path to the database file with feature embedding vectors')

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
