import numpy as np
import cv2
import os
from structlog import get_logger
import model
import uuid
import json
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from tqdm import tqdm
from glob import glob
from utils import pred_to_tl, resize_image

RESIZE_FACTOR = 2

logger = get_logger()
here = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(here, 'results')


class TextblockDetector:

    def __init__(self):
        self.east = None

    def create_model(
        self,
        # checkpoint_path=os.path.join(here, 'weights/EAST_IC15+13_model.h5'),
        checkpoint_path=os.path.join(here, 'east_retrained.h5'),
        is_training=False
    ):

        self.east = model.EAST_model()

        if not checkpoint_path:
            logger.info("Starting with random params")
        else:
            if is_training:
                logger.info(f'Restoring model and optimizer from {checkpoint_path}.')
                # We must load the optimizer aswell
                with open(os.path.join('/'.join(checkpoint_path.split('/')[0:-1]), 'model.json'), 'r') as f:
                    loaded_model_json = f.read()
                self.east.model = model_from_json(loaded_model_json, custom_objects={'tf': tf.compat.v1, 'RESIZE_FACTOR': RESIZE_FACTOR})
                self.east.model.load_weights(checkpoint_path)
            else:
                logger.info(f'Restoring weights from {checkpoint_path}.')
                self.east.model.load_weights(checkpoint_path)

    def predict(self, img):
        """
        :return: {
            'text_lines': [
                {
                    'score': ,
                    'x0': ,
                    'y0': ,
                    'x1': ,
                    ...
                    'y3': ,
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net': ,
                'restore': ,
                'nms': ,
                'cpuinfo': ,
                'meminfo': ,
                'uptime': ,
            }
        }
        """
        result = {}
        result['image_shape'] = img.shape
        print(img.shape)

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        im_resized = im_resized[:, :, ::-1]
        im_resized = (im_resized / 127.5) - 1
        result['working_shape'] = im_resized.shape
        print(im_resized.shape)

        score, geometry = self.east.model.predict(im_resized[np.newaxis, ...])

        text_lines = pred_to_tl(score, geometry, ratio_h, ratio_w)
        print(text_lines)

        result['text_lines'] = text_lines
        return result

    def save_result(self, img, rst):

        def draw_illu(illu, rst):
            for t in rst['text_lines']:
                d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                              t['y2'], t['x3'], t['y3']], dtype='int32')
                d = d.reshape(-1, 2)
                cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
            return illu

        session_id = str(uuid.uuid1())
        dirpath = os.path.join(SAVE_DIR, session_id)
        os.makedirs(dirpath)

        # save input image
        output_path = os.path.join(dirpath, 'input.png')
        cv2.imwrite(output_path, img)

        # save illustration
        output_path = os.path.join(dirpath, 'output.png')
        cv2.imwrite(output_path, draw_illu(img.copy(), rst))

        # save json data
        output_path = os.path.join(dirpath, 'result.json')
        with open(output_path, 'w') as f:
            json.dump(rst, f)

        rst['session_id'] = session_id
        return rst


if __name__ == "__main__":
    tbd = TextblockDetector()
    tbd.create_model(is_training=False,
                     # checkpoint_path="weights/EAST_IC15+13_model.h5"
                     checkpoint_path="train_1s/model-10.h5"
                     )

    # for img in tqdm(glob("/sdata/tb/tb_detector/val/*.jpg")):
    for img in tqdm(glob("/sdata/tb/tb_detector/one_sample/*.jpg")):
        image = cv2.imread(img, 1)
        image = cv2.rotate(image, 2)
        rst = tbd.predict(image)
        res = tbd.save_result(image, rst)
