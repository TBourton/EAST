import os
import argparse
import data_processor
import numpy as np
from tqdm import tqdm
from model import EAST_model
from metrics.metrics import get_metrics
import utils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--nb_workers', type=int, default=4) # number of processes to spin up when using process based threading, as defined in https://keras.io/models/model/#fit_generator
parser.add_argument('--checkpoint_path', type=str, default='tmp/east_resnet_50_rbox') # path to a directory to save model checkpoints during training
parser.add_argument('--restore_model', type=str, default='')
parser.add_argument('--test_data_path', type=str, default='/data/datasets/ICDAR2015/val') # path to validation data
parser.add_argument('--max_image_large_side', type=int, default=1280) # maximum size of the large side of a training image before cropping a patch for training
parser.add_argument('--max_text_size', type=int, default=800) # maximum size of a text instance in an image; image resized if this limit is exceeded
parser.add_argument('--geometry', type=str, default='RBOX') # geometry type to be used; only RBOX is implemented now, but the original paper also uses QUAD
parser.add_argument('--suppress_warnings_and_error_messages', type=bool, default=True) # whether to show error messages and warnings during training (some error messages during training are expected to appear because of the way patches for training are created)
FLAGS = parser.parse_args()


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    east = EAST_model()
    if FLAGS.restore_model != '':
        east.model.load_weights(FLAGS.restore_model)

    east.model.summary()

    tl_true = []
    tl_pred = []

    image_list = np.array(data_processor.get_images(FLAGS.test_data_path))
    print('{} test images in {}'.format(
        image_list.shape[0], FLAGS.test_data_path))
    index = np.arange(0, image_list.shape[0])
    for i in tqdm(index):
        try:
            im_fn = image_list[i]
            im = cv2.imread(im_fn)
            h, w, _ = im.shape
            txt_fn = data_processor.get_text_file(im_fn)
            print(txt_fn)
            if not os.path.exists(txt_fn):
                print('text file {} does not exists'.format(txt_fn))
                continue

            text_polys, text_tags = data_processor.load_annotation(txt_fn)
            text_polys, text_tags = data_processor.check_and_validate_polys(FLAGS, text_polys, text_tags, (h, w))
            im_resized, (ratio_h, ratio_w) = utils.resize_image(im)
            im_resized = im_resized[:, :, ::-1]
            im_resized = (im_resized / 127.5) - 1
            im_resized = np.array(im_resized)
            score, geometry = east.model.predict(im_resized[np.newaxis, ...])
            text_lines = utils.pred_to_tl(score, geometry, ratio_h, ratio_w)

            text_lines, score = utils.tl2raw(text_lines)
            text_polys = utils.flatten_tl(text_polys)

            tl_true.append(text_polys)
            tl_pred.append(text_lines)

            if text_lines is None:
                print("No text lines predicted")
                lines = [""]
            else:
                lines = []
                for i, tl in enumerate(text_lines):
                    line = ""
                    for pt in tl:
                        line += str(int(pt)) + ","
                    line += str(score[i])
                    lines.append(line)
                for i in range(len(lines) - 1):
                    lines[i] += "\n"

            filename = txt_fn.split("/")[-1].split("gt_")[-1]
            filename = "res_" + filename
            with open(os.path.join("/data/tb/tb_test/", filename), 'w') as outfile:
                outfile.writelines(lines)


        except Exception as e:
            print(e)
            raise

    m = get_metrics(tl_true, tl_pred, [])

    print('recall: %.4f, precision: %.4f, hmean: %.4f' % (m['recall'], m['precision'], m['hmean']))


if __name__ == '__main__':
    main()
