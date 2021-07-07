# Based on MediPipe Example Scripts. All rights reserved.
# Modified 2021 Seeed Studio STU, Dmitry Maslov
# SPDX-License-Identifier: MIT

import os
import sys
import time
script_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(script_dir, '..', 'common'))

import cv2
import mediapipe as mp
from tqdm import tqdm
from argparse import ArgumentParser
from cv_utils import init_video_file_capture

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main(args):
    video, video_writer, frame_count = init_video_file_capture(args.video_file_path, 'pose_estimation_demo')
    frame_num = len(frame_count)

    times = []

    with mp_pose.Pose(min_detection_confidence=args.min_detection_confidence, 
                              model_complexity=args.model_complexity,
                              static_image_mode = False) as pose:

      for _ in tqdm(frame_count, desc='Processing frames'):
        frame_present, frame = video.read()
        if not frame_present:
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        start_time = time.time()
        results = pose.process(image)
        end_time = (time.time() - start_time)*1000

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        times.append(end_time)
        video_writer.write(image)

    print('Finished processing frames')
    video.release(), video_writer.release()

    print("Average time(ms): ", sum(times)//frame_num) 
    print("FPS: ", 1000.0 / (sum(times)//frame_num)) # FPS = 1000.0 / average of inference times for all the frames

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_file_path', required=True, type=str,
                        help='Path to the video file to run object detection on')

    parser.add_argument('--min_detection_confidence', default=0.5, type=float,
                        help='Minimum confidence value ([0.0, 1.0]) from the face detection model for the detection to be considered successful. Default to 0.5')

    parser.add_argument('--model_complexity', default=0, type=int,
                        help='Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1')

    args = parser.parse_args()
    main(args)
