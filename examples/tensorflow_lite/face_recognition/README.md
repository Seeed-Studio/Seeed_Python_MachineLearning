# TensorFlow Lite Face Recognition Multi-stage Demo

## Introduction

This demo allows for face recognition from either a video stream or a video file. The face embeddings need to be calculated and saved to a database with calculate_features.py before any of the two examples can be run.

## Prerequisites

Install the dependecnies with
```
pip3 install -r requirements.txt
```
Make sure you have the necessary system packages for OpenCV to work properly.
```
sudo apt-get install libatlas-base-dev libjasper-dev libqtgui4 python3-pyqt5 libqt4-test libilmbase-dev libopenexr-dev libgstreamer1.0-dev libavcodec58 libavformat58 libswscale5
```

## Usage

### Database population

Before we can run face recognition, we need to exctract features from faces we want to recognize and save the features embedding vectors in encoded form in .json file, which serves as a small database. You can do that with calcuate_features.py.

```
python calculate_features.py --help
OpenCV version: 4.5.3
usage: calculate_features.py [-h] --first_stage FIRST_STAGE --second_stage
                             SECOND_STAGE --third_stage THIRD_STAGE
                             [--db_file DB_FILE] --img_file IMG_FILE [--id ID]
                             [--name NAME]

optional arguments:
  -h, --help            show this help message and exit
  --first_stage FIRST_STAGE
                        File path of .tflite file. (default: None)
  --second_stage SECOND_STAGE
                        File path of .tflite file. (default: None)
  --third_stage THIRD_STAGE
                        File path of .tflite file. (default: None)
  --db_file DB_FILE     File path to database (default: database.db)
  --img_file IMG_FILE   File path to picture (default: None)
  --id ID               Path to the video file to run object detection on
                        (default: 0)
  --name NAME           Path to the video file to run object detection on
                        (default: John Doe)
```
For example, to extract a single face embedding vector of Barrack Hussein Obama's face you can run:
```
python calculate_features.py --first_stage ../face_rec_models/YOLOv3_best_recall_quant.tflite --second_stage ../face_rec_models/MobileFaceNet_kpts_quant.tflite --third_stage ../face_rec_models/MobileFaceNet_features_quant.tflite --img_file obama.jpg --name Obama --id 0
```

### Face Recognition from Video File

Once you have a database with at least one face embedding recorded you can try it on a video file, that contains people's faces. Mainly this is used for testing and benchmarking purposes.

Example:
```
python multi_stage_file.py --first_stage ../face_rec_models/YOLOv3_best_recall_quant.tflite --second_stage ../face_rec_models/MobileFaceNet_kpts_quant.tflite --third_stage ../face_rec_models/MobileFaceNet_features_quant.tflite --file ../../sample_files/test_s.mp4
```

### Face Recognition from Video Stream

Finally, for actual application purpose you can use multi_stage_stream.py script. It can get video stream either from OpenCV or picamera, if executed on Raspberry Pi with picamera connected. 

Example:
```
python multi_stage_stream.py --first_stage ../face_rec_models/YOLOv3_best_recall_quant.tflite --second_stage ../face_rec_models/MobileFaceNet_kpts_quant.tflite --third_stage ../face_rec_models/MobileFaceNet_features_quant.tflite
```
The output will be served on a Flask web-server on port 5000. This is done in order to simplify testing and running of an application on headless systems.