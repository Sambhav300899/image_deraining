import numpy as np
np.random.seed(42)

import os
import cv2
from generator import *
from keras_radam import RAdam
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import BatchNormalization, Dropout, Concatenate
from keras.layers import Conv2D, MaxPool2D, Input, Deconvolution2D, Add

class network:
    def __init__(self, model_path = None):

        if model_path == None:
            self.model = self.create_model()
        else:
            self.model = load_model(model_path, custom_objects={'RAdam': RAdam})

    def create_model(self):

        input = Input(shape = (256, 256, 3))

        #encoder
        conv_1 = Conv2D(16, (3, 3), padding = 'same', activation = 'relu')(input)
        pool_1 = MaxPool2D((2, 2))(conv_1)
        conv_2 = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(pool_1)
        pool_2 = MaxPool2D((2, 2))(conv_2)
        conv_3 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(pool_2)
        pool_3 = MaxPool2D((2, 2))(conv_3)
        conv_4 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu')(pool_3)
        pool_4 = MaxPool2D((2, 2))(conv_4)

        #bridge
        conv_b1 = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(pool_4)
        conv_b2 = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(conv_b1)

        #decoder
        deconv_1 = Deconvolution2D(64, (3, 3), subsample = (2, 2), padding = 'same', activation = 'relu')(conv_b2)
        concat_1 = Concatenate()([deconv_1, conv_4])

        conv_6 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu')(concat_1)
        deconv_2 = Deconvolution2D(32, (3, 3), subsample = (2, 2), padding = 'same', activation = 'relu')(conv_6)
        concat_2 = Concatenate()([deconv_2, conv_3])

        conv_7 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(concat_2)
        deconv_3 = Deconvolution2D(16, (3, 3), subsample = (2, 2), padding = 'same', activation = 'relu')(conv_7)
        concat_3 = Concatenate()([deconv_3, conv_2])

        conv_8 = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(concat_3)
        deconv_4 = Deconvolution2D(8, (3, 3), subsample = (2, 2), padding = 'same', activation = 'relu')(conv_8)
        concat_4 = Concatenate()([deconv_4, conv_1])

        conv_9 = Conv2D(16, (3, 3), padding = 'same', activation = 'relu')(concat_4)

        output = Conv2D(3, (1, 1), padding = 'same')(conv_9)

        net = Model(input, output)
        net.summary()
        plot_model(net, 'model.png', show_shapes = True)

        return net

    def train(self, train_path, val_path, rainy_img_path, epochs, bs, lr, callback_dir, log_dir):

        train_gen = generator(train_path, rainy_img_path, bs = int(bs))
        val_gen = generator(val_path, rainy_img_path, bs = int(bs))

        #self.model.compile(loss = 'mae', optimizer = Adam(lr = lr, decay = lr//epochs), metrics = ['mae'])
        #self.model.compile(loss = 'mae', optimizer = 'adadelta', metrics = ['mse'])
        self.model.compile(loss = 'mae', optimizer = RAdam(), metrics = ['mse'])

        filepath = callback_dir + os.path.sep + "weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"

        callbacks = [
        ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only=False),
        TensorBoard(log_dir = log_dir)
        ]

        H = self.model.fit_generator(
        train_gen,
        steps_per_epoch = len(os.listdir(train_path)) // bs,
        validation_data = val_gen,
        validation_steps = len(os.listdir(val_path)) // bs,
        epochs = epochs,
        shuffle = True,
        verbose = 1,
        callbacks = callbacks
        )

        self.model.save('classifier.hdf5')
        self.plot_data(H, epochs)

    def predict(self, img):
        img = np.expand_dims(img, axis = 0)
        preds = self.model.predict(img)
        return preds[0]

    def evaluate(self, img_dir):
        pass

    def plot_data(self, H, n):
        plt.plot(np.arange(0, n), H.history['loss'], label = 'train loss', color = 'g')
        plt.plot(np.arange(0, n), H.history['val_loss'], label = 'validation loss', color = 'b')
        plt.title('training loss and accuracy')
        plt.xlabel('Epoch #')
        plt.ylabel('loss')
        plt.legend(loc = 'upper right')
        plt.savefig('graph_test.png')
