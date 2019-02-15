import cv2

def count_cameras():
    max_tested = 5
    num_found = 0
    for i in range(max_tested):
        print("Probing video device # ", i)
        try:
            temp_camera = cv2.VideoCapture(i)
        except:
            print("Failed ", i)
        if temp_camera.isOpened():
            temp_camera.release()
            print("Camera: ", i)
            num_found += 1
            continue
    return num_found

print("Cameras found: ", count_cameras())
