import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import tensorflow as tf
import pathlib


from tensorflow import keras
from tensorflow.keras import layers, Sequential


data_dir = pathlib.Path("bulk_barn_pics").with_suffix("")
# image_count = len(list(data_dir.glob("*/*.heic")))

# sour_peaches = list(data_dir.glob("sour_peaches/*"))

# Image.open(str(sour_peaches[0])).show()

# Image.open(str("/Users/amir/Downloads/Images.jpeg")).show()


batch_size = 32
img_height = 240
img_width = 320


train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.8,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.8,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names

print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1.0 / 255)


num_classes = len(class_names)

model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
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

model.summary()

# time.sleep(3)
epochs = 100
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Load the image
image_path = "./TestImages/test-3.jpg"  # Change this to the path of your image
image = Image.open(image_path)

# Preprocess the image
# Resize the image to match the input size expected by your model
img_height = 240
img_width = 320
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


# Load the image
image_path = "./TestImages/test-1.jpg"  # Change this to the path of your image
image = Image.open(image_path)

# Preprocess the image
# Resize the image to match the input size expected by your model
img_height = 240
img_width = 320
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
