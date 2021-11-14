import tensorflow as tf




from tensorflow.keras import callbacks
from tensorflow.python.eager.monitoring import Metric
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from my_utils import split_data,create_generators
from deeplearning_models import convolutional_model


if False :
    path_to_data = "D:\DATASETS\INTEL_IMAGE\seg_train"
    path_to_save_train = "D:\DATASETS\INTEL_IMAGE\\training\\train"
    path_to_save_val = "D:\DATASETS\INTEL_IMAGE\\training\\val"

    split_data(path_to_data,path_to_save_train,path_to_save_val)


path_to_train = "D:\DATASETS\INTEL_IMAGE\\training\\train"
path_to_val = "D:\DATASETS\INTEL_IMAGE\\training\\val"
path_to_test = "D:\DATASETS\INTEL_IMAGE\\test"


train, val, test = create_generators(64,path_to_train, path_to_val, path_to_test)

TRAIN = False
TEST = True

if TRAIN:

    epochs = 10
    batch_size = 64

    path_to_save_model = "./Models"
    ckpt_saver = ModelCheckpoint(
        filepath=path_to_save_model,
        monitor="val_loss",
        verbose = 1,
        save_best_only=True,
        save_freq="epoch"
    )

    early_stop = EarlyStopping(
        monitor = "val_loss",
        patience= 6,
        restore_best_weights=True
    )

    model = convolutional_model(train.num_classes)
    model.compile(
        optimizer = "adam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )

    model.fit(
        train,
        validation_data = val,
        epochs = epochs,
        batch_size = batch_size,
        callbacks = [ckpt_saver,early_stop]
    )


if TEST:
    model = tf.keras.models.load_model("./Models")
    model.summary()

    print("evaluating validation set :")
    model.evaluate(val)

    print("evaluating test set :")
    model.evaluate(test)
