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
