# MediaPipe Sample Applications

## Introduction
Google MediaPipe offers ready-to-use yet customizable Python solutions as a prebuilt Python package. 

We provide example scripts for performing inference from video file and video stream with `run_video_file.py` and `run_video_stream.py`. For detailed instructions execute ```run_video_file.py --help``` or ```run_video_stream.py --help```

## Prerequisites

##### PyArmNN

Before proceeding to the next steps, make sure that you have successfully installed the MediaPipe on your system by following the instructions in the README.

You can verify that MediaPipe library is installed using:
```bash
$ pip3 show mediapipe
```

##### Dependencies

Install the following libraries on your system:
```bash
sudo apt install ffmpeg python3-opencv
```

Create a virtual environment:
```bash
python3 -m venv devenv --system-site-packages
source devenv/bin/activate
```

### Python bindings for 32bit version

```
sudo apt install ffmpeg python3-opencv
pip3 install mediapipe-rpi4
```

### Python bindings for 64bit version

Pre-built wheels for Python 3.7 64bit OS were not available at the moment of writing of this article, so we compiled and shared them ourselves.

```
sudo apt install ffmpeg python3-opencv
wget www.files.seeedstudio.com/ml/mediapipe/mediapipe-0.8-cp37-cp37m-linux_aarch64.whl
pip3 install mediapipe-0.8-cp37-cp37m-linux_aarch64.whl
```
