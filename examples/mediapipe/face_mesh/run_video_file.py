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
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from cv_utils import init_video_file_capture

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def main(args):
    video, video_writer, frame_count = init_video_file_capture(args.video_file_path, 'face_mesh_demo')
    frame_num = len(frame_count)
    print(frame_count)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    times = []

    with mp_face_mesh.FaceMesh(min_detection_confidence=args.min_detection_confidence, 
                              min_tracking_confidence=args.min_tracking_confidence) as face_mesh:

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
          results = face_mesh.process(image)
          end_time = (time.time() - start_time)*1000

          # Draw the face mesh annotations on the image.
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
              mp_drawing.draw_landmarks(
                  image=image,
                  landmark_list=face_landmarks,
                  connections=mp_face_mesh.FACE_CONNECTIONS,
                  landmark_drawing_spec=drawing_spec,
                  connection_drawing_spec=drawing_spec)

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
                        help='Path to the first stage model to use')
    parser.add_argument('--min_tracking_confidence', default=0.5, type=float,
                        help='Path to the second stage model to use')

    args = parser.parse_args()
    main(args)
