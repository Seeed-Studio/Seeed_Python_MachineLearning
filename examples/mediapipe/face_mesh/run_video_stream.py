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
mp_face_mesh = mp.solutions.face_mesh

def main(args):
    video = init_video_stream_capture(args.video_source)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(min_detection_confidence=args.min_detection_confidence, 
                              min_tracking_confidence=args.min_tracking_confidence,
                              static_image_mode = False) as face_mesh:

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
            results = face_mesh.process(image)
            end_time = (time.time() - start_time)*1000

            print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
            print("Time(ms): ", (time.time() - start_time)*1000) 

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

            cv2.imshow('MediaPipe Face Mesh Demo', image)

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
    parser.add_argument('--min_tracking_confidence', default=0.5, type=float,
                        help='Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the face landmarks to be considered tracked successfully, or otherwise face detection will be invoked automatically on the next input image.')

    args = parser.parse_args()
    main(args)