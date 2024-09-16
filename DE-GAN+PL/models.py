from tensorflow import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
import scipy.misc
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Concatenate, UpSampling2D, Dropout, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import BinaryAccuracy, MeanAbsoluteError



class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print(f"\nEpoch {epoch+1}: Learning rate is {lr.numpy()}")

print_lr = PrintLR()

def get_optimizer():
    return Adam(learning_rate=1e-5)

generator_optimizer = Adam(learning_rate=1e-5)
discriminator_optimizer = Adam(learning_rate=1e-5)

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def generator_model(pretrained_weights=None, input_size=(256,256,1), biggest_layer=512):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(biggest_layer//2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(biggest_layer//2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.6)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(biggest_layer, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(biggest_layer, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.6)(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    
    model.compile(optimizer=Adam(learning_rate=1e-5))  

    return model

def discriminator_model(input_size=(256, 256, 1)):
    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)

        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
    img_A = Input(input_size)
    img_B = Input(input_size)
    df = 64
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])
    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 4)
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4)
    discriminator = Model([img_A, img_B], validity)
    discriminator.compile(loss='mse', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])

    return discriminator

def build_vgg(gt_shape=(256, 256, 1), layer_name='block3_conv4'):
    input_layer = Input(shape=gt_shape)
    def convert_to_rgb(x):
        return tf.tile(x, [1, 1, 1, 3])
    x = Lambda(convert_to_rgb)(input_layer) if gt_shape[-1] == 1 else input_layer
    vgg = VGG19(weights="imagenet", include_top=False, input_tensor=x)
    model = Model(inputs=input_layer, outputs=vgg.get_layer(layer_name).output)
    return model

global_vgg = build_vgg(gt_shape=(256, 256, 1), layer_name='block3_conv4')
global_vgg.trainable = False

def perceptual_loss_fn(y_true, y_pred):
    y_true_features = global_vgg(y_true)
    y_pred_features = global_vgg(y_pred)
    return MeanSquaredError()(y_true_features, y_pred_features)

def get_gan_network(disc_model, gen_model, input_size=(256, 256, 1)):
    
    disc_model.trainable = False
    gan_input = Input(input_size)
    gen_output = gen_model(gan_input)
    valid = disc_model([gen_output, gan_input])

    valid = Lambda(lambda x: x, name='valid')(valid)
    gen_output = Lambda(lambda x: x, name='gen_output')(gen_output)

    gan = Model(inputs=gan_input, outputs=[valid, gen_output, gen_output], name="GAN_Model")

    gan.compile(
        optimizer=Adam(learning_rate=1e-5, clipvalue=1.0),  

        loss=['mse', 'binary_crossentropy', perceptual_loss_fn],
        loss_weights=[1, 50, 100],
        metrics={
            'valid': [BinaryAccuracy()],
            'gen_output': [None, None]
        }
    )
    
    return gan

