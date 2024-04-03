import time
#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras.models import load_model
import pathlib
import tensorflow as tf

class_names = ['sour_cola_belt', 'sour_keys', 'sour_peaches', 'sour_rainbow_belt', 'sour_variety']

model = load_model('./model.keras')
# Load the image
image = Image.open("./TestImages/test-5.jpg")

# Image dimentions model will accept (will resisze if it is lower than this)
img_height = 600
img_width = 600
image = image.resize((img_width, img_height))

# Convert the image to a NumPy array and normalize the pixel values
image_array = np.array(image) / 255.0
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