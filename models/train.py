
"""
Construct a custom generator to train or test deep learning model

"""

import os
import glob
import random
import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf
from metimage.utils.ImageProcessing import SplitTiles
from metimage.models.ResNet import ResNet_v2

def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def translate(img, x, y):
    img_x, img_y = img.shape
    if x > 0:
        image_trans = np.delete(np.r_[np.zeros([x, img_y]), img], [range(img_x, (x + img_x))], axis=0)
    else:
        image_trans = np.delete(np.r_[img, np.zeros([-x, img_y])], [range(0, -x)], axis=0)
    if y > 0:
        image_trans = np.delete(np.c_[np.zeros([img_x, y]), image_trans], [range(img_y, (y + img_y))], axis=1)
    else:
        image_trans = np.delete(np.c_[image_trans, np.zeros([img_x, -y])], [range(0, -y)], axis=1)
    return image_trans

class MultiChannel_generator_v3(tf.keras.utils.Sequence):
    """
    A multichannel generator for binary classification.
    """

    def __init__(self, _dir_,
                 batch_size,
                 Sampling_list=None,
                 Sampling_number=None,
                 pixelx=224, pixely=224, overlap_col=0, overlap_row=0,
                 shuffle=True, seed=42, Print=False, Normalization=True,
                 Augmentation=False,RT_shift=20,MZ_shift=10,Int_shift=[0.1,10]):
        label_dir = os.listdir(_dir_)
        NEGList = glob.glob(_dir_ + "/" + label_dir[0] + "/*." + "npz")
        POSList = glob.glob(_dir_ + "/" + label_dir[1] + "/*." + "npz")
        self.xList = NEGList + POSList
        self.yList = [0] * len(NEGList) + [1] * len(POSList)
        randnum = 42
        random.Random(randnum).shuffle(self.xList)
        random.Random(randnum).shuffle(self.yList)
        self.batch_size = batch_size
        self.Sampling_list = Sampling_list
        self.Sampling_number = Sampling_number
        self.shuffle = shuffle
        self.seed = seed
        self.Print = Print
        self.Normalization = Normalization
        self.Augmentation = Augmentation
        self.pixelx = pixelx
        self.pixely = pixely
        self.overlap_col = overlap_col
        self.overlap_row = overlap_row
        self.RT_shift = RT_shift
        self.MZ_shift = MZ_shift
        self.Int_shift = Int_shift
        self.n = len(self.xList)
        self.n_name = pd.Series(self.xList).nunique()
        self.n_type = pd.Series(self.yList).nunique()
    def on_epoch_end(self):
        if self.shuffle:
            randnum = random.randint(0, 100)
            random.seed(randnum)
            random.shuffle(self.xList)
            random.seed(randnum)
            random.shuffle(self.yList)
    def __get_input(self, path):
        SparseTable = sparse.load_npz(path)
        RawImage = SparseTable.todense()
        # Normalization part
        if self.Normalization:
            RawImage = np.array(RawImage)
            RawImage = RawImage/RawImage.max()
        if self.Augmentation:
            if self.RT_shift is not 0:
                x = random.sample(range(-self.RT_shift, self.RT_shift), 1)[0]
            if self.MZ_shift is not 0:
                y = random.sample(range(-self.MZ_shift, self.MZ_shift), 1)[0]
            RawImage = translate(RawImage, x, y)
            # Update from published version. Old version code:
            # C = random.sample(range(1, 100), 1)[0]*0.1
            C = random.uniform(self.Int_shift[0],self.Int_shift[1])
            RawImage = RawImage*C
        RawImage = pd.DataFrame(RawImage)
        tiles = SplitTiles(RawImage, pixelx=self.pixelx, pixely=self.pixely,
                           overlap_col=self.overlap_col, overlap_row=self.overlap_row)
        if self.Sampling_number is not None:
            random.seed(self.seed)
            sample = random.sample(range(tiles.shape[2]), self.Sampling_number)
        else:
            sample = self.Sampling_list
        if self.Print:
            print(path)
        return tiles[:, :, sample]

    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_data(self, batches):
        X_batch = np.asarray([self.__get_input(path=x) for x in self.xList[batches]])
        Y_batch = np.asarray(self.yList[batches])
        return X_batch, Y_batch

    def __getitem__(self, index):
        batches = slice(index * self.batch_size, (index + 1) * self.batch_size)
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

def LoadModel(Tiles_no, weight_dir):
    _input_shape = (224, 224, Tiles_no)
    model = ResNet_v2(18) #Use Fixed
    model.build((None,) + _input_shape)
    model.summary()
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=METRICS)
    model.load_weights(weight_dir)
    return model

import pickle
def ParseIndex(lst_dir):
    f = open(lst_dir,
             "rb")
    Sampling_list = pickle.load(f)
    return Sampling_list

from tensorflow.keras.optimizers import Adam
def MakeModel(Tiles_no):
    """
    Build and compile deep learning model.
    """
    _input_shape = (224, 224, Tiles_no)
    model = ResNet_v2(18)
    model.build((None,) + _input_shape)
    model.summary()
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=METRICS)
    return model

from tensorflow.keras.callbacks import EarlyStopping
def train(train_dir, test_dir, batch_size,model, Sampling_list, seed=42, save_dir=".", epochs=200, workers=6,
          Augmentation=True,RT_shift=20,MZ_shift=10,Int_shift=[0.1,10],
          Normalization=False,
          pixelx=224, pixely=224, overlap_col=0, overlap_row=0,
          save_during_training=False, save_epoch=10,
          earlystopping=False, min_delta=0, patience=20,
          save_best_model=True):
    """
    Train deep learning model.
    """

    train_generator = MultiChannel_generator_v3(train_dir, batch_size=batch_size, Sampling_list=Sampling_list,
                                                Sampling_number=None, seed=seed,Normalization=Normalization,
                                                pixelx=pixelx, pixely=pixely,
                                                overlap_col=overlap_col, overlap_row=overlap_row,
                                                Print=False,shuffle=False,
                                                Augmentation=Augmentation,RT_shift=RT_shift,MZ_shift=MZ_shift,
                                                Int_shift=Int_shift)
    validation_generator = MultiChannel_generator_v3(test_dir, batch_size=1, Sampling_list=Sampling_list,
                                                     Sampling_number=None, seed=seed,Normalization=Normalization,
                                                     pixelx=pixelx, pixely=pixely,
                                                     overlap_col=overlap_col, overlap_row=overlap_row,
                                                     Print=False,shuffle=False,Augmentation=False)
    callbacks_list = []
    log_dir = save_dir +"/" + "log"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list.append(tensorboard_callback)
    if save_during_training:
        checkpoint_path = save_dir+"/"+"cp-{epoch:04d}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True,
                                                         save_freq=save_epoch*batch_size)
        model.save_weights(checkpoint_path.format(epoch=0))
        callbacks_list.append(cp_callback)
    if earlystopping:
        earlystop = EarlyStopping(monitor='val_loss', mode='min', min_delta=min_delta, patience=patience,verbose=1)
        callbacks_list.append(earlystop)
    if save_best_model:
        best_weights_filepath = save_dir+"/best_weights.hdf5"
        saveBestModel = tf.keras.callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss',
                                                           verbose=1, save_weights_only=True, save_best_only=True,
                                                           mode='auto')
        callbacks_list.append(saveBestModel)

    model.fit(train_generator,epochs=epochs, verbose=1, validation_data=validation_generator,validation_freq=1,
              workers=workers, use_multiprocessing=True, callbacks=callbacks_list)
    model.save_weights(save_dir + '/model_weight.h5')
    return model

def Model_evaluation(test_dir, model, Sampling_list, Normalization=False,
                     pixelx=224, pixely=224, overlap_col=0, overlap_row=0,
                     workers=12):
    """
    Test performce of deep learning model with testing set.

    """
    test_generator = MultiChannel_generator_v3(test_dir, batch_size=1, Sampling_list=Sampling_list,
                                               Sampling_number=None, seed=42, Print=False,shuffle=False,
                                               Normalization=Normalization,
                                               pixelx=pixelx, pixely=pixely,
                                               overlap_col=overlap_col, overlap_row=overlap_row,
                                               )
    model.evaluate(test_generator, workers=workers, use_multiprocessing=True)

from tqdm import tqdm
def Model_prediction(pred_dir, model, Sampling_list, Normalization=False,
                     pixelx=224, pixely=224, overlap_col=0, overlap_row=0,
                     print=True,save_dir=".",
                     workers=12):
    """
    Prediction of unlabelled samples.

    """
    rawlist = glob.glob(pred_dir+ "/**/*.npz",recursive=True)
    pre_res_all = []
    sample_name = []
    for file in tqdm(rawlist):
        sample_name.append(os.path.basename(file))
        SparseTable = sparse.load_npz(file)
        RawImage = pd.DataFrame(SparseTable.todense())
        if Normalization:
            RawImage = np.array(RawImage)
            RawImage = RawImage / RawImage.max()
        tiles = SplitTiles(RawImage, pixelx=pixelx, pixely=pixely, overlap_col=overlap_col, overlap_row=overlap_row)
        tiles = tiles[:, :, Sampling_list]
        input = tiles[np.newaxis, :, :, :]
        pre_res_all.append(model.predict(input, workers=workers, use_multiprocessing=True)[0][0])
    pred_class = [1 if i>0.5 else 0 for i in pre_res_all]
    dict = {"name": sample_name, "filename": rawlist, "prediction": pre_res_all, "pred_class": pred_class}
    if print:
        data = pd.DataFrame(dict)
        data.to_csv(save_dir+"/prediction.csv")
    return dict


# def test(test_dir,weight_dir, Sampling_list,method = "Prediction", Normalization=False, workers=12):
#     """
#     Test performce of deep learning model.
#
#     :param: method: method of model test
#                     Prediction: return prediction probability value.
#                     Evaluation: return model performance.
#
#     """
#     test_generator = MultiChannel_generator_v3(test_dir, batch_size=1, Sampling_list=Sampling_list,
#                                                Sampling_number=None, seed=42, Print=False,shuffle=False,
#                                                Normalization=Normalization)
#     Tiles_no = len(Sampling_list)
#     model = LoadModel(Tiles_no, weight_dir)
#     if method == "Evaluation":
#         model.evaluate(test_generator,workers=workers, use_multiprocessing=True)
#     else:
#         predict_res_class = model.predict_class(test_generator, workers=workers, use_multiprocessing=True)
#         predict_res_prob = model.predict(test_generator, workers=workers, use_multiprocessing=True)
#         xList = test_generator.xList
#         yList = test_generator.yList
#         dict = {"xlist": xList, "ylist": yList, "prediction_class": predict_res_class, "Prob": predict_res_prob}
#         return dict

