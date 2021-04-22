import tensorflow as tf 
print(tf.__version__)

import tarfile
import urllib.request
import os

#모델 다운로드 하고 압축 푸는 코드

MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu_8'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
MODELS_DIR = 'data/models'

#모델 다운로드 받을 수 있는 곳 google에 tf zoo 검색
# http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz

MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
MODELS_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME

PATH_TO_MODEL_TAR = os.path.join('data/models' ,MODEL_TAR_FILENAME)
PATH_TO_CHPT = os.path.join('data/models', os.path.join(MODEL_NAME , 'checkpoint/'))
PATH_TO_CFG = os.path.join('data/models', os.path.join(MODEL_NAME , 'pipeline.comfig'))


# 모델받아서 압축풀기

if not os.path.exists(PATH_TO_CKPT):
    print('Downloading model. This may take a while... ', end='')
    urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
    tar_file = tarfile.open(PATH_TO_MODEL_TAR)
    tar_file.extractall(MODELS_DIR)
    tar_file.close()
    os.remove(PATH_TO_MODEL_TAR)
    print('Done')

# 레이블 다운로드 받기

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
if not os.path.exists(PATH_TO_LABELS):
    print('Downloading label file... ', end='')
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
    print('Done')




