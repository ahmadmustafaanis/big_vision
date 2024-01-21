import pickle
import sys

sys.path.append("/home/ahmad/Desktop/projects/big_vision/")

import cv2
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

from big_vision.models.proj.clippo.one_tower import Model as OneTowerModel


# Assuming DenseModel is defined in your code
class DenseModel(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.num_classes)(x)


# Function to load the model and its weights
def load_model(model_class, weights_path, num_classes=None):
    model = model_class(num_classes=num_classes) if num_classes else model_class()
    with open(weights_path, "rb") as f:
        params = pickle.load(f)
    return model, params


# Load the models
model1, model1_params = load_model(OneTowerModel, "model1_weights.pkl")
model2, model2_params = load_model(DenseModel, "model2_weights.pkl", num_classes=10)


# Preprocess the image (this function should be the same as used during training)
def preprocess_image(image):
    # image = cv2.imread(image_path)
    # convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # convert the image to a batch of size 1
    processed_image = np.expand_dims(image, axis=0)
    return processed_image


# Inference function
def infer(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Convert to jax.numpy array
    processed_image = jnp.array(processed_image)

    # Apply the first model
    logits, *_ = model1.apply({"params": model1_params}, processed_image)
    # Apply the second model
    predictions = model2.apply({"params": model2_params}, logits)
    # print(predictions)

    # Convert to class labels
    predicted_class = np.argmax(predictions, axis=-1)
    return predicted_class


ds = tfds.load("mnist", split="test")

num = 0
for example in ds.take(10):
    image = example["image"].numpy()
    label = example["label"].numpy()

    predicted_class = infer(image)
    print("Predicted class:", predicted_class)
    print("True class:", label)
    if num == 10:
        break
    num += 1
