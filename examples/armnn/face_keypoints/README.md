# PyArmNN Face keypoint detection Sample Application

## Introduction
This sample application guides the user and shows how to perform face keypoint detection using PyArmNN API. 

The application takes a model and video file or camera feed as input, runs inference on each frame, and draws bounding boxes around detected faces and five keypoints(left eye, right eye, nose, left corner of the mouth, right corner of the mouth) with the corresponding labels and confidence scores overlaid.

## Face keypoint detection from Video File
Face keypoint detection demo that takes a video file, runs inference on each frame producing
bounding boxes and five keypoints on detected faces, and saves the processed video.

Example usage:

```bash
python3 run_video_file.py --first_model_file_path YOLO_best_mAP.tflite --second_model_file MobileFaceNet_kpts.tflite --video_file_path ../samples/test_s.mp4 
```

## Face keypoint detection from Video Stream

Face keypoint detection demo that takes a video file, takes a video stream from a device, runs inference
on each frame producing bounding boxes and five keypoints on detected faces, and displays a window with the latest processed frame.

Example usage:

```bash
DISPLAY=:0 python3 run_video_stream.py --first_model_file_path YOLO_best_mAP.tflite --second_model_file MobileFaceNet_kpts.tflite
```

This application has been verified to work against the YOLOv2 detection layer MobileNet models and MobileFaceNet keypoints detector, which can be downloaded from:

https://files.seeedstudio.com/ml/keypoint_detection_models.zip
