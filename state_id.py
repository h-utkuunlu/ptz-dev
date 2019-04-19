"Identification state as part of the PTZ tracker finite state machine"
from torchvision.transforms import Compose
from torch import cat
from numpy import argmax
from fastai.vision import *

from dnn import real_time_evaluate, read_stats, Normalize, Resize, ToTensor
from utils import expand_bbox

data_prep = Compose(
    [Resize(224),
     Normalize(read_stats("./dataset_stats")),
     ToTensor()])


def in_id_fn(system):

    transformed_ims = [data_prep(img) for img in system.cur_imgs]
    # transformed_ims = [Image(data_prep(img)) for img in system.cur_imgs]

    predictions = real_time_evaluate(system.network, cat(transformed_ims))
    # predictions = real_time_evaluate_fastai(system.network, transformed_ims)

    drone = False
    max_pred_i = argmax(predictions)
    max_pred = predictions[max_pred_i]

    if max_pred >= system.detect_thresh:
        system.drone_bbox = system.cur_bboxes[max_pred_i]
        system.fsm.drone()
        drone = True
        print("Drone identified")

    if not drone:
        system.fsm.not_drone()


def out_id_fn(system):
    pass


def async_id(system, frame):

    vals = [int(a) for a in system.drone_bbox]
    vals = [
        max(0, vals[0]),
        max(0, vals[1]),
        min(1920, vals[2]),
        min(1080, vals[3])
    ]

    x, y, w, h = expand_bbox(*vals)

    roi = frame[y:y + h, x:x + w].copy()
    system.update_gui(async_frame=roi)

    prediction = real_time_evaluate(system.network, data_prep(roi))[0]
    if prediction >= system.detect_thresh:
        return 1
    else:
        return 0


def async_id_fastai(system, frame):

    vals = [int(a) for a in system.drone_bbox]
    vals = [
        max(0, vals[0]),
        max(0, vals[1]),
        min(1920, vals[2]),
        min(1080, vals[3])
    ]

    x, y, w, h = expand_bbox(*vals)

    roi = frame[y:y + h, x:x + w].copy()
    system.update_gui(async_frame=roi)

    prediction = real_time_evaluate_fastai(system.network,
                                           Image(data_prep(roi)))[0]
    if prediction == 1:
        return 1
    else:
        return 0
