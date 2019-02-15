# import the necessary packages
import argparse
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument("skip", help="Save every Nth frame")
args = parser.parse_args()

skip = int(args.skip)

def extract_frames(path):
    """
    :type path: string
    :return None
    """
    # grab a pointer to the video file
    video = cv2.VideoCapture(path)
    video_name = path.rsplit(".", 1)[0]  # take only the video name without prefix
    # capture first frame, and keep a frame count
    success, frame = video.read()
    count = 1
    while success:
        if count % skip == 1:
            cv2.imwrite("{}({}).jpeg".format(video_name, count), frame)
        success, frame = video.read()
        count += 1
    video.release()

def main():
    videos = [file for file in os.listdir() if file.endswith(".mp4")]

    for video_path in videos:
        extract_frames(video_path)

if __name__ == '__main__':
    main()
