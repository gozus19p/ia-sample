import os
import re
import shutil
import string

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses


model = tf.keras.models.load_model("./model")

# model.predict(x=["The movie was good", "The movie was bad", "The movie was quite good"])

