import time
import cv2
import cv2.aruco as aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters_create()

def in_id_fn(parent):
    #print('id.', end='')

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
    
    # print("No drone :(")
    # print(parent.state)
    parent.not_drone()

def out_id_fn(parent):
    pass

def async_id(parent):
    
    frame = parent.camera.cvreader.Read()
    if frame is None:
        print("None frame")
        return

    vals = [int(a) for a in parent.drone_bbox]
    x, y, w, h = max(0, vals[0]), max(0, vals[1]), min(parent.camera.width, vals[2]), min(parent.camera.height, vals[3]) 

    roi = frame[y:y+h, x:x+w].copy()
    
    cv2.namedWindow("async_id", cv2.WINDOW_NORMAL)
    
    cv2.imshow("async_id", roi)
    cv2.waitKey(1)

    _, ids, _ = aruco.detectMarkers(roi, aruco_dict, parameters=parameters)

    if ids is not None and ids[0][0] == 42:
        return 1
    else:
        return 0
