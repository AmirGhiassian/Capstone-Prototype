import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras.layers import Input

ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pathlib


from tensorflow import keras
from tensorflow.keras import layers, Sequential


def predict_image(image_path):
    # Load the image
    # Change this to the path of your image
    image = Image.open(image_path)

    # Preprocess the image
    # Resize the image to match the input size expected by your model
    img_height = 1280
    img_width = 720
    image = image.resize((img_width, img_height))

    # Convert the image to a NumPy array and normalize the pixel values
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(image_array)

    # Interpret the results
    # Assuming your model outputs logits, you might want to apply softmax to get probabilities
    probabilities = tf.nn.softmax(predictions[0])
    # Get the predicted class index
    predicted_class = np.argmax(probabilities)

    # Print the predicted class and corresponding probability
    print("Predicted Class:", class_names[predicted_class])
    print("Confidence:", probabilities[predicted_class].numpy())


data_dir = pathlib.Path("bulk_barn_pics").with_suffix("")
# image_count = len(list(data_dir.glob("*/*.heic")))

# sour_peaches = list(data_dir.glob("sour_peaches/*"))

# Image.open(str(sour_peaches[0])).show()

# Image.open(str("/Users/amir/Downloads/Images.jpeg")).show()


batch_size = 32
img_height = 1280
img_width = 720


train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,  # adjusted for 70-30 split
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,  # consistent with the training split (70-30)
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

# Define the data augmentation model
data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.2),
    ]
)


# Set up the early stopping callback
# early_stopping = EarlyStopping(
#     monitor="val_loss",  # You can change this to 'val_accuracy' if you care more about accuracy
#     patience=5,  # Number of epochs to wait after min has been hit. After this number of epochs without improvement, training stops.
#     verbose=1,
#     mode="min",  # 'min' because we want to minimize the loss; for accuracy, use 'max'.
#     restore_best_weights=True,  # This rolls back the model weights to those of the epoch with the best value of the monitored metric.
# )

normalization_layer = layers.Rescaling(1.0 / 255)

num_classes = len(class_names)


model = Sequential(
    [
        # Data Augmentation layers
        data_augmentation,
        Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        # Convolutional layers
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


# time.sleep(3)
epochs = 100
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


predict_image("./TestImages/test-1.jpg")
predict_image("./TestImages/test-2.jpg")
predict_image("./TestImages/test-3.jpg")

# data_augmentation = keras.Sequential(
#     [
#         layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
#         layers.RandomRotation(0.1),
#         layers.RandomZoom(0.1),
#     ]
# )


# model = Sequential(
#     [
#         data_augmentation,
#         layers.Rescaling(1.0 / 255),
#         layers.Conv2D(16, 3, padding="same", activation="relu"),
#         layers.MaxPooling2D(),
#         layers.Conv2D(32, 3, padding="same", activation="relu"),
#         layers.MaxPooling2D(),
#         layers.Conv2D(64, 3, padding="same", activation="relu"),
#         layers.MaxPooling2D(),
#         layers.Dropout(0.2),
#         layers.Flatten(),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(num_classes, name="outputs"),
#     ]
# )


# model.compile(
#     optimizer="adam",
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )


# model.summary()

# epochs = 300
# history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
