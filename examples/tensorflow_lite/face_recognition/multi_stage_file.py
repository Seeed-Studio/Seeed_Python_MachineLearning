import argparse
import cv2
import numpy as np
import skimage 
import skimage.transform
import json, base64
import time
from tqdm import tqdm

from cv_utils import decode_yolov3, preprocess, init_video_file_capture
from tflite_runtime.interpreter import Interpreter

FACE_ANCHORS = [[[0.51424575, 0.54116074], [0.29523918, 0.45838044], [0.21371929, 0.21518053]],
               [[0.10255913, 0.42572159], [0.05785894, 0.17925645], [0.01839256, 0.07238193]]]

IMG_SHAPE = (128, 128) # in HW form 
offset_x = 0
offset_y = -15
src = np.array([(44+offset_x, 59+offset_y),
                (84+offset_x, 59+offset_y),
                (64+offset_x, 82+offset_y),
                (47+offset_x, 105),
                (81+offset_x, 105)], dtype=np.float32)

def read_db(db_path = 'database.db'):
    try:
        f = open(db_path, 'r')
    except FileNotFoundError:
        clear_db(db_path)
        f = open(db_path, 'r')

    content = f.read()
    #print(content)
    if content:
        db = json.loads(content)
    f.close()
    return db

def clear_db(db_path = 'database.db'):

    f = open(db_path,'w')
    db = {}
    content = json.dumps(db)
    f.write(content)
    f.close()

def draw_bounding_boxes(frame, detections, kpts, ids):

    def _to_original_scale(boxes, frame_height, frame_width):
        minmax_boxes = np.empty(shape=(4, ), dtype=np.int)

        cx = boxes[0] * frame_width
        cy = boxes[1] * frame_height
        w = boxes[2] * frame_width
        h = boxes[3] * frame_height
        
        minmax_boxes[0] = cx - w/2
        minmax_boxes[1] = cy - h/2
        minmax_boxes[2] = cx + w/2
        minmax_boxes[3] = cy + h/2

        return minmax_boxes

    color = (0, 255, 0)
    label_color = (125, 125, 125)

    for i in range(len(detections)):
        _, box, _ = [d for d in detections[i]]

        # Obtain frame size and resized bounding box positions
        frame_height, frame_width = frame.shape[:2]

        x_min, y_min, x_max, y_max = _to_original_scale(box, frame_height, frame_width)
        # Ensure box stays within the frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

        # Draw bounding box around detected object
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Create label for detected object class
        label = 'ID: {} Name: {} {}%'.format(*ids[i])
        label_color = (255, 255, 255)

        # Make sure label always stays on-screen
        x_text, y_text = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0][:2]

        lbl_box_xy_min = (x_min, y_min if y_min<25 else y_min - y_text)
        lbl_box_xy_max = (x_min + int(0.75 * x_text), y_min + y_text if y_min<25 else y_min)
        lbl_text_pos = (x_min + 5, y_min + 16 if y_min<25 else y_min - 5)

        # Add label and confidence value
        cv2.rectangle(frame, lbl_box_xy_min, lbl_box_xy_max, color, -1)
        cv2.putText(frame, label, lbl_text_pos, cv2.FONT_HERSHEY_DUPLEX, 0.70, label_color, 1, cv2.LINE_AA)

        for kpt_set in kpts:
            for kpt in kpt_set:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 5, (255, 0, 0), 2)

def process_faces(frame, detections, db):
    kpts_list = []
    id_list = []

    def _to_original_scale(boxes, frame_height, frame_width):
        minmax_boxes = np.empty(shape=(4, ), dtype=np.int)

        cx = boxes[0] * frame_width
        cy = boxes[1] * frame_height
        w = boxes[2] * frame_width
        h = boxes[3] * frame_height
        
        minmax_boxes[0] = cx - w/2
        minmax_boxes[1] = cy - h/2
        minmax_boxes[2] = cx + w/2
        minmax_boxes[3] = cy + h/2

        return minmax_boxes

    for i in range(len(detections)):
        _, box, _ = [d for d in detections[i]]

        # Obtain frame size and resized bounding box positions
        frame_height, frame_width = frame.shape[:2]

        x_min, y_min, x_max, y_max = _to_original_scale(box, frame_height, frame_width)
        # Ensure box stays within the frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)

        x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

        face_img = frame[y_min:y_max, x_min:x_max]

        plist = second_stage_network.run(face_img)[0]

        le = (x + int(plist[0] * w+5), y + int(plist[1] * h+5))
        re = (x + int(plist[2] * w), y + int(plist[3] * h+5))
        n = (x + int(plist[4] * w), y + int(plist[5] * h))
        lm = (x + int(plist[6] * w), y + int(plist[7] * h))
        rm = (x + int(plist[8] * w), y + int(plist[9] * h))
        kpts = [le, re, n, lm, rm]
        kpts_list.append(kpts)
        kpts = np.array(kpts, dtype = np.float32)

        transformer = skimage.transform.SimilarityTransform() 
        transformer.estimate(kpts, src) 
        M = transformer.params[0: 2, : ] 
        warped_img = cv2.warpAffine(frame, M, (IMG_SHAPE[1], IMG_SHAPE[0]), borderValue = 0.0) 

        features = third_stage_network.run(warped_img)[0]

        highest_score = 0

        for id in db.keys():
            cos_sim = np.dot(features, db[id]['vector'])/(np.linalg.norm(features)*np.linalg.norm(db[id]['vector']))
            cos_sim /= 2
            cos_sim += 0.5
            cos_sim *= 100
            if highest_score < cos_sim:
                highest_score = cos_sim
                recognized_id = id
                
        if highest_score > 70.0:
            print(recognized_id, db[recognized_id]['name'], highest_score)
            id_list.append([recognized_id, db[recognized_id]['name'], highest_score])
        else:
            id_list.append(['X', '', 0.0])
    return kpts_list, id_list

class NetworkExecutor(object):

    def __init__(self, model_file):

        self.interpreter = Interpreter(model_file, num_threads=3)
        self.interpreter.allocate_tensors()
        _, self.input_height, self.input_width, _ = self.interpreter.get_input_details()[0]['shape']
        self.tensor_index = self.interpreter.get_input_details()[0]['index']

    def get_output_tensors(self):

      output_details = self.interpreter.get_output_details()
      tensor_indices = []
      tensor_list = []

      for output in output_details:
            tensor = np.squeeze(self.interpreter.get_tensor(output['index']))
            tensor_list.append(tensor)

      return tensor_list

    def run(self, image):
        if image.shape[1:2] != (self.input_height, self.input_width):
            img = cv2.resize(image, (self.input_width, self.input_height))
        img = preprocess(img)
        self.interpreter.set_tensor(self.tensor_index, img)
        self.interpreter.invoke()
        return self.get_output_tensors()

def main(args):
    video, video_writer, frame_count = init_video_file_capture(args.file, 'age_gender_demo')

    frame_num = len(frame_count)
    times = []

    for _ in tqdm(frame_count, desc='Processing frames'):
        frame_present, frame = video.read()
        if not frame_present:
            continue

        start_time = time.time()
        
        results = first_stage_network.run(frame)
        detections = decode_yolov3(netout = results, nms_threshold = 0.1,
                                  threshold = args.threshold, anchors = FACE_ANCHORS)
        kpts, ids = process_faces(frame, detections, db)

        elapsed_ms = (time.time() - start_time) * 1000

        draw_bounding_boxes(frame, detections, kpts, ids)
        times.append(elapsed_ms)
        video_writer.write(frame)

    print('Finished processing frames')
    video.release(), video_writer.release()

    print("Average time(ms): ", sum(times)//frame_num) 
    print("FPS: ", 1000.0 / (sum(times)//frame_num)) # FPS = 1 / time to process loop

if __name__ == "__main__" :

    print("OpenCV version: {}".format(cv2. __version__))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--first_stage', help='Path to the YOLOv3 face detection model to use.', required=True)
    parser.add_argument('--second_stage', help='Path to the keypoints detection model to use.', required=True)
    parser.add_argument('--third_stage', help='Path to the feature vector embedding extractor model to use.', required=True)  

    parser.add_argument('--db_file', help='File path to database', default="database.db") 
    
    parser.add_argument('--threshold', help='Confidence threshold.', default=0.7)
    parser.add_argument('--file', help='File path of video file', required=True)
    args = parser.parse_args()

    first_stage_network = NetworkExecutor(args.first_stage)
    second_stage_network = NetworkExecutor(args.second_stage)
    third_stage_network = NetworkExecutor(args.third_stage)

    db = read_db(args.db_file)
    for item in db:
        db[item]['vector'] = np.frombuffer(base64.b64decode(db[item]['vector']), np.float32)

    main(args)
    
