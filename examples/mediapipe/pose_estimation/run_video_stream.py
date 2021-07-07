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
from argparse import ArgumentParser
from cv_utils import init_video_stream_capture

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main(args):
    video = init_video_stream_capture(args.video_source)

    with mp_pose.Pose(min_detection_confidence=args.min_detection_confidence, 
                              model_complexity=args.model_complexity,
                              static_image_mode = False) as pose:

        while True:

            frame_present, frame = video.read()
            if not frame_present:
                raise RuntimeError('Error reading frame from video stream')

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            start_time = time.time()
            results = pose.process(image)
            end_time = (time.time() - start_time)*1000

            print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
            print("Time(ms): ", (time.time() - start_time)*1000) 

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('MediaPipe Pose Estimation Demo', image)

            if cv2.waitKey(1) == 27:
                print('\nExit key activated. Closing video...')
                break

    video.release(), cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_source', type=int, default=0,
                        help='Device index to access video stream. Defaults to primary device camera at index 0')

    parser.add_argument('--min_detection_confidence', default=0.5, type=float,
                        help='Minimum confidence value ([0.0, 1.0]) from the face detection model for the detection to be considered successful. Default to 0.5')

    parser.add_argument('--model_complexity', default=0, type=int,
                        help='Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1')

    args = parser.parse_args()
    main(args)