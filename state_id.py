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
    #max_pred_i = argmax(predictions)
    #max_pred = predictions[max_pred_i]
    candidates = []

    for i, pred in enumerate(predictions):
        if pred > system.detect_thresh:
            candidates.append((pred, system.cur_bboxes[i]))
    candidates.sort(reverse=True)
    candidates = candidates[:20]

    if len(candidates) == 0:
        system.fsm.not_drone()

    else:
        itered_contours = [i[1] for i in candidates]
        prev_itered_contours = []
        while itered_contours != prev_itered_contours and len(
                itered_contours) > 1:
            prev_itered_contours = list(itered_contours)
            itered_contours = []
            prev_c = prev_itered_contours[0]
            for c in prev_itered_contours[1:]:
                merged_rect = (min(c[0], prev_c[0]), min(c[1], prev_c[1]),
                               max(c[0] + c[2], prev_c[0] + prev_c[2]) -
                               min(c[0], prev_c[0]),
                               max(c[1] + c[3], prev_c[1] + prev_c[3]) -
                               min(c[1], prev_c[1]))
                x, y, w, h = merged_rect

                if prev_c[2] + c[2] >= w and prev_c[3] + c[3] >= h:
                    itered_contours.append(merged_rect)
                    prev_c = merged_rect
                else:
                    prev_c = c

        areas = list(map(lambda x: x[2] * x[3], itered_contours))
        system.drone_bbox = itered_contours[areas.index(max(areas))]
        system.fsm.drone()
        print("Drone identified")


def out_id_fn(system):
    pass


def area(rect):
    return rect[2] * rect[3]


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
