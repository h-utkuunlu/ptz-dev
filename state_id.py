import time
import cv2
#import cv2.aruco as aruco
from dnn import real_time_evaluate, PrepareRTImage, read_stats

#aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
#parameters = aruco.DetectorParameters_create()

data_prep = PrepareRTImage(224, read_stats("./dataset_stats"))

def in_id_fn(parent):

    drone = False
    predictions = real_time_evaluate(parent.network, data_prep(parent.cur_imgs))
    for iter, pred in enumerate(predictions):
        if pred == 1:
            parent.drone_bbox = parent.cur_bboxes[iter]
            parent.drone()
            drone = True
            print("Drone identified")
            break
        else:
            pass

    '''
    for i, img in enumerate(parent.cur_imgs):
        _, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

        if ids is not None and ids[0][0] == 42:
            parent.drone_bbox = parent.cur_bboxes[i]
            # print(parent.state)
            parent.drone()
            print("drone")
            return
        else:
            pass #print("not_drone")
    '''
    if not drone:
        parent.not_drone()

def out_id_fn(parent):
    pass

def async_id(parent):
    
    frame = parent.camera.cvreader.Read()
    if frame is None:
        return

    vals = [int(a) for a in parent.drone_bbox]
    x, y, w, h = max(0, vals[0]), max(0, vals[1]), min(parent.camera.width, vals[2]), min(parent.camera.height, vals[3]) 

    roi = frame[y:y+h, x:x+w].copy()
    
    cv2.namedWindow("async_id", cv2.WINDOW_NORMAL)
    cv2.imshow("async_id", roi)
    cv2.waitKey(1)

    prediction = real_time_evaluate(parent.network, data_prep([roi]))[0]
    if prediction == 1:
        return 1
    else:
        return 0

    '''
    _, ids, _ = aruco.detectMarkers(roi, aruco_dict, parameters=parameters)

    if ids is not None and ids[0][0] == 42:
        return 1
    else:
        return 0
    '''
