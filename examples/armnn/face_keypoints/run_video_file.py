# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# Modified 2021 Seeed Studio STU, Dmitry Maslov
# SPDX-License-Identifier: MIT

"""
Face keypoint detection demo that takes a video file, runs inference on each frame producing
bounding boxes and five keypoints on detected faces, and saves the processed video.

python3 run_video_file.py --first_model_file_path YOLO_best_mAP.tflite --second_model_file MobileFaceNet_kpts.tflite --video_file_path ../samples/test_s.mp4 

"""

import os
import sys
import time
script_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(script_dir, '..', 'common'))

import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from yolov2 import yolo_processing, yolo_resize_factor
from utils import dict_labels
from cv_utils import init_video_file_capture, resize_with_aspect_ratio, preprocess, preprocess_array
from network_executor import ArmnnNetworkExecutor
import pyarmnn as ann

def process_faces(frame, detections, executor_kp, resize_factor):
    kpts_list = []

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
        face_img = preprocess_array(face_img)

        input_tensors = ann.make_input_tensors([executor_kp.input_binding_info], [face_img])

        plist = executor_kp.run(input_tensors)[0][0]

        le = (x + int(plist[0] * w+5), y + int(plist[1] * h+5))
        re = (x + int(plist[2] * w), y + int(plist[3] * h))
        n = (x + int(plist[4] * w), y + int(plist[5] * h))
        lm = (x + int(plist[6] * w), y + int(plist[7] * h))
        rm = (x + int(plist[8] * w), y + int(plist[9] * h))
        kpts = [le, re, n, lm, rm]

        kpts_list.append(kpts)

    return kpts_list

def draw_result(frame: np.ndarray, detections: list, resize_factor, kpts):
    """
    Draws bounding boxes around detected objects and adds a label and confidence score.

    Args:
        frame: The original captured frame from video source.
        detections: A list of detected objects in the form [class, [box positions], confidence].
        resize_factor: Resizing factor to scale box coordinates to output frame size.
        kpts: List containing information about face keypoints in format [[le, re, n, lm, rm], [le, re, n, lm, rm], ...]
    """
    for i in range(len(detections)):
        class_idx, box, confidence = [d for d in detections[i]]
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

        for kpt in kpts[i]:
            cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 5, (0, 0, 255), 5)


def main(args):
    video, video_writer, frame_count = init_video_file_capture(args.video_file_path, 'face_keypoint_demo')
    frame_num = len(frame_count)

    executor_fd = ArmnnNetworkExecutor(args.first_model_file_path, args.preferred_backends)
    executor_kp = ArmnnNetworkExecutor(args.second_model_file_path, args.preferred_backends)    

    process_output, resize_factor = yolo_processing, yolo_resize_factor(video, executor_fd.input_binding_info)

    times = []

    for _ in tqdm(frame_count, desc='Processing frames'):
        frame_present, frame = video.read()
        if not frame_present:
            continue

        input_tensors = preprocess(frame, executor_fd.input_binding_info)

        start_time = time.time() # start time of the loop
        output_result = executor_fd.run(input_tensors)
        detections = process_output(output_result)
        kpts = process_faces(frame, detections, executor_kp, resize_factor)
        end_time = (time.time() - start_time)*1000

        draw_result(frame, detections, resize_factor, kpts)
        times.append(end_time)
        video_writer.write(frame)

    print('Finished processing frames')
    video.release(), video_writer.release()

    print("Average time(ms): ", sum(times)//frame_num) 
    print("FPS: ", 1000.0 / (sum(times)//frame_num)) # FPS = 1 / time to process loop

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_file_path', required=True, type=str,
                        help='Path to the video file to run object detection on')

    parser.add_argument('--first_model_file_path', required=True, type=str,
                        help='Path to the first stage model to use')
    parser.add_argument('--second_model_file_path', required=True, type=str,
                        help='Path to the second stage model to use')

    parser.add_argument('--preferred_backends', type=str, nargs='+', default=['CpuAcc', 'CpuRef'],
                        help='Takes the preferred backends in preference order, separated by whitespace, '
                             'for example: CpuAcc GpuAcc CpuRef. Accepted options: [CpuAcc, CpuRef, GpuAcc]. '
                             'Defaults to [CpuAcc, CpuRef]')
    args = parser.parse_args()
    main(args)
