import time
import cv2
from torchvision.transforms import Compose
from torch import cat
from dnn import real_time_evaluate, read_stats, Normalize, Resize, ToTensor
from utils import expand_bbox

data_prep = Compose([ Resize(224), Normalize(read_stats("./dataset_stats")), ToTensor() ])

def in_id_fn(parent):

    drone = False
    transformed_ims = [data_prep(i) for i in parent.cur_imgs]
    predictions = real_time_evaluate(parent.network, cat(transformed_ims))

    for iter, pred in enumerate(predictions):
        if pred == 1:
            parent.drone_bbox = parent.cur_bboxes[iter]
            parent.drone()
            drone = True
            print("Drone identified")
            break
        else:
            pass

    if not drone:
        parent.not_drone()

def out_id_fn(parent):
    pass

def async_id(parent):
    
    frame = parent.camera.cvreader.Read()
    if frame is None:
        return

    vals = [int(a) for a in parent.drone_bbox]
    vals = [max(0, vals[0]), max(0, vals[1]), min(1920, vals[2]), min(1080, vals[3])]

    x, y, w, h = expand_bbox(*vals)

    roi = frame[y:y+h, x:x+w].copy()
    parent.gui.update(async_frame = roi)
    
    prediction = real_time_evaluate(parent.network, data_prep(roi))[0]
    if prediction == 1:
        return 1
    else:
        return 0
