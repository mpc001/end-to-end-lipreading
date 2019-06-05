"""
Transforms mp4 videos to npz and crop fixed mouth ROIs. Code has strong assumptions on the dataset organization!

"""

import os
import cv2
import glob
import numpy as np


def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video[...,::-1]


basedir = ''
basedir_to_save = ''
filenames = glob.glob(os.path.join(basedir, '*', '*', '*.mp4'))
for filename in filenames:
    data = extract_opencv(filename)[:, 115:211, 79:175]
    path_to_save = os.path.join(basedir_to_save,
                                filename.split('/')[-3],
                                filename.split('/')[-2],
                                filename.split('/')[-1][:-4]+'.npz')
    if not os.path.exists(os.path.dirname(path_to_save)):
        try:
            os.makedirs(os.path.dirname(path_to_save))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    np.savez( path_to_save, data=data)
