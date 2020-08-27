import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import argparse
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import tensorflow.keras.backend as K
from utils import pred_to_tl, tl2raw
from model import EAST_model
from losses import dice_loss, rbox_loss
import data_processor
from metrics.metrics import get_metrics

tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, default=512) # input size for training of the network
parser.add_argument('--batch_size', type=int, default=4) # batch size for training
parser.add_argument('--nb_workers', type=int, default=4) # number of processes to spin up when using process based threading, as defined in https://keras.io/models/model/#fit_generator
parser.add_argument('--init_learning_rate', type=float, default=0.0001) # initial learning rate
parser.add_argument('--lr_decay_rate', type=float, default=0.94) # decay rate for the learning rate
parser.add_argument('--lr_decay_steps', type=int, default=130) # number of steps after which the learning rate is decayed by decay rate
parser.add_argument('--max_epochs', type=int, default=800) # maximum number of epochs
parser.add_argument('--gpu_list', type=str, default='1') # list of gpus to use
parser.add_argument('--checkpoint_path', type=str, default='tmp/east_resnet_50_rbox') # path to a directory to save model checkpoints during training
parser.add_argument('--save_checkpoint_epochs', type=int, default=10) # period at which checkpoints are saved (defaults to every 10 epochs)
parser.add_argument('--restore_model', type=str, default='')
parser.add_argument('--training_data_path', type=str, default='../data/ICDAR2015/train_data') # path to training data
parser.add_argument('--validation_data_path', type=str, default='../data/MLT/val_data_latin') # path to validation data
parser.add_argument('--max_image_large_side', type=int, default=1280) # maximum size of the large side of a training image before cropping a patch for training
parser.add_argument('--max_text_size', type=int, default=800) # maximum size of a text instance in an image; image resized if this limit is exceeded
parser.add_argument('--min_text_size', type=int, default=10) # minimum size of a text instance; if smaller, then it is ignored during training
parser.add_argument('--min_crop_side_ratio', type=float, default=0.1) # the minimum ratio of min(H, W), the smaller side of the image, when taking a random crop from thee input image
parser.add_argument('--geometry', type=str, default='RBOX') # geometry type to be used; only RBOX is implemented now, but the original paper also uses QUAD
parser.add_argument('--suppress_warnings_and_error_messages', type=bool, default=False) # whether to show error messages and warnings during training (some error messages during training are expected to appear because of the way patches for training are created)
FLAGS = parser.parse_args()

gpus = list(range(len(FLAGS.gpu_list.split(','))))


class CustomModelCheckpoint(Callback):
    def __init__(self, model, path, period, save_weights_only):
        super(CustomModelCheckpoint, self).__init__()
        self.period = period
        self.path = path
        # We set the model (non multi gpu) under an other name
        self.model_for_saving = model
        self.epochs_since_last_save = 0
        self.save_weights_only = save_weights_only

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_weights_only:
                self.model_for_saving.save_weights(self.path.format(epoch=epoch + 1, **logs), overwrite=True)
            else:
                self.model_for_saving.save(self.path.format(epoch=epoch + 1, **logs), overwrite=True)


class SmallTextWeight(Callback):
    def __init__(self, weight):
        self.weight = weight

    # TO BE CHANGED
    def on_epoch_end(self, epoch, logs={}):
        #K.set_value(self.weight, np.minimum(epoch / (0.5 * FLAGS.max_epochs), 1.))
        K.set_value(self.weight, 0)

class ValidationEvaluator(Callback):
    def __init__(self, validation_data, validation_data_length, period=5):
        super(Callback, self).__init__()

        self.period = period
        self.validation_data = validation_data
        self.validation_data_length = validation_data_length

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.period == 0:
            #images = self.validation_data[0]
            #overly_small_text_region_training_masks = self.validation_data[1]
            #text_region_boundary_training_masks = self.validation_data[2]
            #score_maps = self.validation_data[3]
            #geo_maps = self.validation_data[4]

            #val_loss, val_score_map_loss, val_geo_map_loss = self.model.evaluate([images, overly_small_text_region_training_masks, text_region_boundary_training_masks, score_maps],
            #                                                                     [score_maps, geo_maps],
            #                                                                     batch_size=FLAGS.batch_size)
            score_maps, geo_maps = [], []
            pred_score_maps, pred_geo_maps = [], []
            ratios = []

            for i, batch in enumerate(self.validation_data, 1):
                if i > int(np.floor(self.validation_data_length / FLAGS.batch_size)):
                    break
                X, y = batch
                y_pred = self.model.predict(X)
                pred_score_maps.append(y_pred[0])
                pred_geo_maps.append(y_pred[1])
                score_maps.append(y[0])
                geo_maps.append(y[1])
                ratios.append(y[2])

            score_maps = np.concatenate(score_maps)
            geo_maps = np.concatenate(geo_maps)
            pred_score_maps = np.concatenate(pred_score_maps)
            pred_geo_maps = np.concatenate(pred_geo_maps)
            ratios = np.concatenate(ratios)
            try:
                m = compute_metrics(score_maps, geo_maps, pred_score_maps, pred_geo_maps, ratios)
            except Exception as e:
                print(f"Error computing metrics: {e}")
                import traceback
                if not FLAGS.suppress_warnings_and_error_messages:
                    traceback.print_exc()
                m = {'precision': 0, "hmean": 0, 'recall': 0, 'numGlobalCareGt': 0, 'matchedSum': 0, 'numGlobalCareDet': 0}

            # print('\nEpoch %d: val_loss: %.4f, val_score_map_loss: %.4f, val_geo_map_loss: %.4f, recall: %.4f, precision: %.4f, hmean: %.4f' % (epoch + 1, val_loss, val_score_map_loss, val_geo_map_loss, m['recall'], m['precision'], m['hmean']))
            print('\nEpoch %d: recall: %.4f, precision: %.4f, hmean: %.4f, raw_gt: %d, raw_pred: %d, raw_match: %d' % (epoch + 1, m['recall'], m['precision'], m['hmean'], m['numGlobalCareGt'], m['numGlobalCareDet'], m['matchedSum']))


def lr_decay(epoch):
    return FLAGS.init_learning_rate * np.power(FLAGS.lr_decay_rate, epoch // FLAGS.lr_decay_steps)


def compute_metrics(true_score_maps, true_geo_maps, pred_score_maps, pred_geo_maps, ratios):
    tl_pred = []
    tl_true = []
    scores = []
    total_pred = 0
    total_true = 0

    # The true/predicted values are given/computed on the same resized image
    for i in range(len(true_score_maps)):
        ratio_h, ratio_w = ratios[i][0], ratios[i][1]
        # Map the true scores and geos to ordered dict format text_lines
        t = pred_to_tl(true_score_maps[i], true_geo_maps[i], ratio_h, ratio_w)
        t, score = tl2raw(t)  # Convert the ordered dict form to raw format
        tl_true.append(t)
        if t:
            total_true += len(t)

        # Map the predictions to ordered dict format text_lines
        t = pred_to_tl(pred_score_maps[i], pred_geo_maps[i], ratio_h, ratio_w)
        t, score = tl2raw(t)  # Convert the ordered dict form to raw format
        tl_pred.append(t)
        if t:
            total_pred += len(t)
        scores.append(score)

    print(f"\nNum tl pred {total_pred}, Num tl true {total_true}.")
    return get_metrics(tl_true, tl_pred, scores)


def main(argv=None):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    # check if checkpoint path exists
    if not os.path.exists(FLAGS.checkpoint_path):
        os.mkdir(FLAGS.checkpoint_path)
    else:
        #if not FLAGS.restore:
        #    shutil.rmtree(FLAGS.checkpoint_path)
        #    os.mkdir(FLAGS.checkpoint_path)
        shutil.rmtree(FLAGS.checkpoint_path)
        os.mkdir(FLAGS.checkpoint_path)

    # train_data_generator = data_processor.generator(FLAGS)
    # train_samples_count = data_processor.count_samples(FLAGS)
    train_data_generator = data_processor.DataGenerator(FLAGS)
    # train_data_generator = data_processor.DataGenerator(FLAGS)
    # val_data = data_processor.load_data(FLAGS)
    val_data = data_processor.val_generator(FLAGS)
    val_samples_count = data_processor.count_val_samples(FLAGS)

    east = EAST_model(FLAGS.input_size)
    if FLAGS.restore_model != '':
        east.model.load_weights(FLAGS.restore_model)

    score_map_loss_weight = K.variable(0.01, name='score_map_loss_weight')

    small_text_weight = K.variable(0., name='small_text_weight')

    lr_scheduler = LearningRateScheduler(lr_decay)
    ckpt = CustomModelCheckpoint(model=east.model, path=FLAGS.checkpoint_path + '/model-{epoch:02d}.h5', period=FLAGS.save_checkpoint_epochs, save_weights_only=False)
    small_text_weight_callback = SmallTextWeight(small_text_weight)
    validation_evaluator = ValidationEvaluator(val_data, val_samples_count)
    callbacks = [lr_scheduler, ckpt, small_text_weight_callback, validation_evaluator]

    opt = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=FLAGS.init_learning_rate, name='AdamW')

    east.model.compile(
        loss=[dice_loss(east.overly_small_text_region_training_mask, east.text_region_boundary_training_mask, score_map_loss_weight, small_text_weight),
              rbox_loss(east.overly_small_text_region_training_mask, east.text_region_boundary_training_mask, small_text_weight, east.target_score_map)],
        loss_weights=[1., 1.],
        optimizer=opt,
        # metrics=[['accuracy'], ['accuracy']]
    )
    east.model.summary()

    model_json = east.model.to_json()
    with open(FLAGS.checkpoint_path + '/model.json', 'w') as json_file:
        json_file.write(model_json)

    # history = east.model.fit(train_data_generator, epochs=FLAGS.max_epochs, steps_per_epoch=train_samples_count/FLAGS.batch_size, workers=FLAGS.nb_workers, use_multiprocessing=True, max_queue_size=10, callbacks=callbacks, verbose=1)
    history = east.model.fit(train_data_generator, epochs=FLAGS.max_epochs, workers=FLAGS.nb_workers, use_multiprocessing=True, max_queue_size=10, callbacks=callbacks, verbose=1)

    east.model.save('east_retrained.h5')

if __name__ == '__main__':
    main()
