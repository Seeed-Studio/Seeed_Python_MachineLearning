#!/usr/bin/env python

import cv2
import os
import sys, getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner

show_camera = True

def draw_result(frame, class_name, bb, confidence):
    """
    Draws bounding boxes around detected objects and adds a label and confidence score.
    Args:
        frame: The original captured frame from video source.
        detections: A list of detected objects in the form [class, [box positions], confidence].
        resize_factor: Resizing factor to scale box coordinates to output frame size.
        face_data: List containing information about age and gender
    """
    color = (255, 0, 0)

    x_min, y_min, x_max, y_max = bb['x'], bb['y'], bb['x']+ bb['width'], bb['y']+ bb['height']

    # Draw bounding box around detected object
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    # Create label for detected object class
    label = "{}, {}".format(class_name, confidence)
    label_color = (255, 255, 255)

    # Make sure label always stays on-screen
    x_text, y_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0][:2]

    lbl_box_xy_min = (x_min, y_min if y_min<25 else y_min - y_text)
    lbl_box_xy_max = (x_min + int(0.75 * x_text), y_min + y_text if y_min<25 else y_min)
    lbl_text_pos = (x_min + 5, y_min + 16 if y_min<25 else y_min - 5)

    # Add label and confidence value
    cv2.rectangle(frame, lbl_box_xy_min, lbl_box_xy_max, color, -1)
    cv2.putText(frame, label, lbl_text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.70, label_color, 1, cv2.LINE_AA)


def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    def get_path(model_name):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        modelfile = os.path.join(dir_path, model_name)
        print('MODEL: ' + modelfile)
        return modelfile

    detection_model = get_path(args[0])
    classification_model = get_path(args[1])

    with ImageImpulseRunner(detection_model) as detection_runner, ImageImpulseRunner(classification_model) as classification_runner:

        detection_model_info = detection_runner.init()
        classification_model_info = classification_runner.init()

        print('Loaded detection model runner for "' + detection_model_info['project']['owner'] + ' / ' + detection_model_info['project']['name'] + '"')
        detection_labels = detection_model_info['model_parameters']['labels']

        print('Loaded detection model runner for "' + classification_model_info['project']['owner'] + ' / ' + classification_model_info['project']['name'] + '"')
        classification_labels = classification_model_info['model_parameters']['labels']

        class_model_input_height = classification_model_info['model_parameters']['image_input_height']
        class_model_input_width = classification_model_info['model_parameters']['image_input_width']

        if len(args)>= 3:
            videoCaptureDeviceId = int(args[2])
        else:
            port_ids = get_webcams()
            if len(port_ids) == 0:
                raise Exception('Cannot find any webcams')
            if len(args)<= 1 and len(port_ids)> 1:
                raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
            videoCaptureDeviceId = int(port_ids[0])

        camera = cv2.VideoCapture(videoCaptureDeviceId)

        ret = camera.read()[0]
        if ret:
            backendName = camera.getBackendName()
            w = camera.get(3)
            h = camera.get(4)
            print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
            camera.release()
        else:
            raise Exception("Couldn't initialize selected camera.")

        for det_res, img in detection_runner.classifier(videoCaptureDeviceId):
            print('Found %d bounding boxes (%d ms.)' % (len(det_res["result"]["bounding_boxes"]), det_res['timing']['dsp'] + det_res['timing']['classification']))
            for bb in det_res["result"]["bounding_boxes"]:
                print('%s (%.2f): x=%d y=%d w=%d h=%d\n' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))

                cropped_img = img[bb['y']:bb['y']+bb['height'], bb['x']:bb['x']+bb['width']]
                resized_img = cv2.resize(cropped_img, (class_model_input_width, class_model_input_height))

                features, cropped = classification_runner.get_features_from_image(resized_img)

                # the image will be resized and cropped, save a copy of the picture here
                # so you can see what's being passed into the classifier
                #cv2.imwrite('debug.jpg', cropped)

                class_res = classification_runner.classify(features)

                if "classification" in class_res["result"].keys():
                    print('Classification result (%d ms.) \n' % (class_res['timing']['dsp'] + class_res['timing']['classification']), end='')
                    top_score = 0
                    top_label = ''

                    for label in classification_labels:
                        score = class_res['result']['classification'][label]
                        print('%s: %.2f\n' % (label, score), end='')
                        if score >= top_score:
                            top_score = score
                            top_label = label

                    print('----------------------\n', flush=True)
                    print('Top result: %s with confidence %.2f\n' % (top_label, top_score), end='')
                    print('----------------------\n', flush=True)

                draw_result(img, top_label, bb, top_score)

            if (show_camera):
                cv2.imshow('edgeimpulse', img)
                if cv2.waitKey(1) == ord('q'):
                    break

        detection_runner.stop()
        classification_runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])