"""
Transforms mp4 audio to npz. Code has strong assumptions on the dataset organization!

"""

import os
import glob
import librosa
import numpy as np


basedir = ''
basedir_to_save = ''
filenames = glob.glob(os.path.join(basedir, '*', '*', '*.mp4'))
for filename in filenames:
    data = librosa.load(filename, sr=16000)[0][-19456:]
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
