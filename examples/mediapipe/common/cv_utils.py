# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# Modified 2021 Seeed Studio STU, Dmitry Maslov
# SPDX-License-Identifier: MIT

"""
This file contains helper functions for reading video/image data and
 pre/postprocessing of video/image data using OpenCV.
"""

import os

import cv2
import numpy as np

def count_frames_manual(video):
    # initialize the total number of frames read
    total = 0
    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()
        
        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break
        # increment the total number of frames read
        total += 1
    # return the total number of frames in the video file
    return total

def create_video_writer(video: cv2.VideoCapture, video_path: str, name: str):
    """
    Creates a video writer object to write processed frames to file.

    Args:
        video: Video capture object, contains information about data source.
        video_path: User-specified video file path.
        output_path: Optional path to save the processed video.

    Returns:
        Video writer object.
    """
    _, ext = os.path.splitext(video_path)

    i, filename = 0, os.path.join(str(), f'{name}{ext}')

    while os.path.exists(filename):
        i += 1
        filename = os.path.join(str(), f'{name}({i}){ext}')
    print(filename)
    video_writer = cv2.VideoWriter(filename=filename,
                                   fourcc=get_source_encoding_int(video),
                                   fps=int(video.get(cv2.CAP_PROP_FPS)),
                                   frameSize=(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                              int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    return video_writer


def init_video_file_capture(video_path: str, name: str):
    """
    Creates a video capture object from a video file.

    Args:
        video_path: User-specified video file path.
        output_path: Optional path to save the processed video.

    Returns:
        Video capture object to capture frames, video writer object to write processed
        frames to file, plus total frame count of video source to iterate through.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video file not found for: {video_path}')

    video = cv2.VideoCapture(video_path)
    if not video.isOpened:
        raise RuntimeError(f'Failed to open video capture from file: {video_path}')

    video_writer = create_video_writer(video, video_path, name)

    iter_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    return video, video_writer, range(iter_frame_count)


def init_video_stream_capture(video_source: int):
    """
    Creates a video capture object from a device.

    Args:
        video_source: Device index used to read video stream.

    Returns:
        Video capture object used to capture frames from a video stream.
    """
    video = cv2.VideoCapture(video_source)
    if not video.isOpened:
        raise RuntimeError(f'Failed to open video capture for device with index: {video_source}')
    print('Processing video stream. Press \'Esc\' key to exit the demo.')
    return video

def get_source_encoding_int(video_capture):
    return int(video_capture.get(cv2.CAP_PROP_FOURCC))
