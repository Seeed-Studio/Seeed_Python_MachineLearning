# PyArmNN Face recognition Sample Application

## Introduction
This sample application guides the user and shows how to perform face recognition using PyArmNN API. 

The application takes three models and video file or camera feed as input, runs inference on each frame producing bounding boxes and ID numbers corresponding to entries in database.

## Database population

Before we can run face recognition, we need to exctract features from faces we want to recognize and save the features embedding vectors in encoded form in .json file, which serves as a small database. You can do that with calcuate_features.py.

Example usage:

```bash
python3 calculate_features.py --fd_model_file_path ../face_rec_models/YOLOv2_best_mAP.tflite --kp_model_file_path ../face_rec_models/MobileFaceNet_kpts.tflite --fe_model_file_path ../face_rec_models/MobileFaceNet_features.tflite --db_file_path database.db --id 0 --name Paul --picture_file_path paul.png
```

## Face recognition from Video File
Face recognition demo that takes a video file, runs inference on each frame producing
bounding boxes and ID numbers corresponding to entries in database, and saves the processed video.

Example usage:

```bash
python3 run_video_file.py --video_file_path test_s.mp4 --db_file_path database.db --fd_model_file_path ../face_rec_models/YOLOv2_best_mAP.tflite --kp_model_file_path ../face_rec_models/MobileFaceNet_kpts.tflite --fe_model_file_path ../face_rec_models/MobileFaceNet_features.tflite 
```

## Face recognition from Video Stream

Face recognition demo that takes a video stream from a device, runs inference
on each frame producing bounding boxes and ID numbers corresponding to entries in database,
and displays a window with the latest processed frame.

Example usage:

```bash
DISPLAY=:0 python3 run_video_stream.py --db_file_path database.db --fd_model_file_path ../face_rec_models/YOLOv2_best_mAP.tflite --kp_model_file_path ../face_rec_models/MobileFaceNet_kpts.tflite --fe_model_file_path ../face_rec_models/MobileFaceNet_features.tflite 
```

This application has been verified to work against the YOLOv2 detection layer MobileNet models, MobileFaceNet keypoints detector and MobileFaceNet face feature embedding extractor which can be downloaded from:

WIP
