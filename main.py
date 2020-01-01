from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from models import CNNArchitecture

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# To regenerate the results
seed = 7
np.random.seed(seed)


# Tensorflow Stuff
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def train_model(model, callbacks_list, name, train_batchs, test_batchs, epochs):

    num_train_images = len(train_batchs.filenames)
    train_batch_size = train_batchs.batch_size
    num_test_images = len(test_batchs.filenames)
    test_batch_size = test_batchs.batch_size

    # Start Training
    history = model.fit_generator(
        train_batches,
        validation_data=test_batches,
        epochs=epochs,
        verbose=1,
        # steps_per_epoch=num_train_images // train_batch_size,
        steps_per_epoch=10,
        # validation_steps=num_test_images // test_batch_size,
        validation_steps=5,
        callbacks=callbacks_list
    )

    _, (ax1, ax2) = plt.subplots(2)
    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
    ax1.set_title(f'{name}_model_accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title(f'{name}_model_loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig(f"./model_checkpoints/{name}/{name}.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--processed_data', default='./updatedData/', help='Path to processed dataset in the format of ImageDataGenerator'
    )
    parser.add_argument(
        '--epochs', default=50, help='Define the number of epochs for the training of the model'
    )

    # Get Processed data
    train_path = parser.parse_args().processed_data + "train"
    test_path = parser.parse_args().processed_data + "test"

    # Load the data in batches via ImageDataGenerator
    train_batches = ImageDataGenerator(rescale=1/255).flow_from_directory(
        train_path, target_size=(512, 512), classes=['valid', 'invalid'], batch_size=8)
    test_batches = ImageDataGenerator(rescale=1/255).flow_from_directory(
        test_path, target_size=(512, 512), classes=['valid', 'invalid'], batch_size=4)

    # Meta Data
    input_shape = train_batches.image_shape
    num_classes = len(np.unique(train_batches.classes))
    loss = 'categorical_crossentropy'
    optimizer = Adam()
    epochs = int(parser.parse_args().epochs)

    # Get model architecure
    cnn_model = CNNArchitecture(input_shape, num_classes, loss, optimizer)

    # Train VGG
    vgg_16, callbacks_list, name = cnn_model.VGG_16()
    train_model(vgg_16, callbacks_list, name,
                train_batches, test_batches, epochs)

    # Train Resnet
    # resnet_50, callbacks_list, name = cnn_model.resnet_50()
    # train_model(resnet_50, callbacks_list, name,
    #             train_batches, test_batches, epochs)

    # Train Inception
    # inception_v3, callbacks_list, name = cnn_model.inception_v3()
    # train_model(inception_v3, callbacks_list,
    #             name, train_batches, test_batches, epochs)

    # Train Inception_Resnet_V2
    # inception_resnet_v2, callbacks_list, name = cnn_model.inception_resnet_v2()
    # train_model(inception_resnet_v2, callbacks_list,
    #             name, train_batches, test_batches, epochs)

    # Train Xception
    # xception, callbacks_list, name = cnn_model.xception()
    # train_model(xception, callbacks_list, name,
    # train_batches, test_batches, epochs)

    # # Train DeXpression
    # deXpression, callbacks_list, name = cnn_model.DeXpression()
    # train_model(deXpression, callbacks_list, name,
    #             train_batches, test_batches, epochs)
