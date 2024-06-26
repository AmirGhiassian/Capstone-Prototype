import time
#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pathlib
from keras.applications.resnet50 import ResNet50

from tensorflow import keras
from tensorflow.keras import layers, Sequential

model = ""
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img('./TestImages/test-4.jpg', target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.array([img_array])
    predictions = model.predict(img_array)
    print(predictions)

    class_id = np.argmax(predictions, axis = 1)
    print(class_id)

    print(class_names[class_id.item()]) 
with tf.device('/GPU:0'):
    data_dir = pathlib.Path("bulk_barn_pics").with_suffix("")
    # image_count = len(list(data_dir.glob("*/*.heic")))

    # sour_peaches = list(data_dir.glob("sour_peaches/*"))

    # Image.open(str(sour_peaches[0])).show()

    # Image.open(str("/Users/amir/Downloads/Images.jpeg")).show()


    batch_size = 16
    img_height = 1280
    img_width = 720


    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,  # adjusted for 70-30 split
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,  # consistent with the training split (70-30)
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    print(class_names)

    # configuring dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# ---------------------------------------------
    if(not pathlib.Path("./model.h5").is_file()):
        
       #  Define the data augmentation model
        data_augmentation = Sequential(
            [
                RandomFlip("horizontal"),
                RandomRotation(0.1),
                RandomZoom(0.2),
            ]
        )

        # early_stopping = EarlyStopping(
        #     monitor="val_loss",  # You can change this to 'val_accuracy' if you care more about accuracy
        #     patience=5,  # Number of epochs to wait after min has been hit. After this number of epochs without improvement, training stops.
        #     verbose=1,
        #     mode="min",  # 'min' because we want to minimize the loss; for accuracy, use 'max'.
        #     restore_best_weights=True,  # This rolls back the model weights to those of the epoch with the best value of the monitored metric.
        # )

        normalization_layer = layers.Rescaling(1.0 / 255)

        num_classes = len(class_names)
        print(num_classes)
        base_model = ResNet50(include_top=False,
                      weights='imagenet',
                      input_shape=(img_height, img_width, 3))
        base_model.trainable = False  # Freeze the convolutional base
        model = Sequential([
 base_model,
    layers.GlobalAveragePooling2D(),
layers.Dense(512, activation='relu'),
layers.Dropout(0.5),
layers.Dense(256, activation='relu'),
layers.Dropout(0.5),
layers.Dense(128, activation='relu'),
layers.Dropout(0.5),
layers.Dense(64, activation='relu'),
layers.Dropout(0.5),
layers.Dense(num_classes, activation='softmax'),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        normalization_layer = layers.Rescaling(1.0 / 255)

        # time.sleep(3)
        epochs = 200


        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        
        model.save("model.keras")
        predict_image("./TestImages/test-1.jpg", model)
        predict_image("./TestImages/test-3.jpg", model)
        
    else:
        model = load_model('./model.keras')
       
