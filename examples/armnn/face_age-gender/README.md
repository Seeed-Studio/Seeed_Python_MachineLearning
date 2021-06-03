# PyArmNN Human face age/gender recognition Sample Application

## Introduction
This sample application guides the user and shows how to perform age/gender recognition using PyArmNN API. 

The application takes a model and video file or camera feed as input, runs inference on each frame, and draws bounding boxes around detected faces and age/gender labels overlaid.

## Human face age/gender recognitionfrom Video File
Human face age/gender recognition demo that takes a video file, runs inference on each frame producing
bounding boxes and labels around detected faces, and saves the processed video.

Example usage:

```bash
python3 run_video_file.py --first_model_file_path YOLO_best_mAP.tflite --second_model_file MobileNet-v1-age-gender.tflite --video_file_path ../samples/test_s.mp4 
```

## Human face age/gender recognition from Video Stream

Human face age/gender recognition demo that takes a video stream from a device, runs inference
on each frame producing bounding boxes and labels around detected faces,
and displays a window with the latest processed frame.

Example usage:

```bash
DISPLAY=:0 python3 run_video_stream.py --first_model_file_path YOLO_best_mAP.tflite --second_model_file MobileNet-v1-age-gender.tflite
```

This application has been verified to work against the YOLOv2 detection layer MobileNet models and MobileFaceNet keypoints detector, which can be downloaded from:

https://files.seeedstudio.com/ml/age_gender_recognition_models.zip
