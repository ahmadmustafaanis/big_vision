import pickle
import sys

sys.path.append("/content/big_vision")

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
from flax.training import common_utils, train_state
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from big_vision.configs.proj.clippo import train_clippo
from big_vision.models.proj.clippo.one_tower import Model as OneTowerModel
from big_vision.models.proj.clippo.one_tower import load as one_tower_load
from big_vision.models.vit import load as vit_load

num_classes = 101
num_epochs = 10
batch_size = 128
# Define cross entropy loss
def cross_entropy(logits, labels):
    one_hot = common_utils.onehot(labels, num_classes=num_classes)
    xent = optax.softmax_cross_entropy(logits, one_hot)
    return jnp.mean(xent)




class DenseModel(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.num_classes)(x)


ds = tfds.load("food101_updated_text_no_img_preprocessing")

# Initialize the first model (frozen)
model1 = OneTowerModel()
init_params = vit_load(
    None,
    "/content/clippo_b16_yfcc100m_i21k_init_75c4.npz",
    None,
)


# Initialize the second model (trainable)
model2 = DenseModel(num_classes=num_classes)
initial_params_model2 = model2.init(jax.random.PRNGKey(0), jnp.ones((1, 768)))["params"]


# Define the TrainState for the second model
class TrainState(train_state.TrainState):
    model1_params: any = None


train_state = TrainState.create(
    apply_fn=model2.apply,
    params=initial_params_model2,
    tx=optax.adam(1e-3),
    model1_params=init_params,
)


def compute_metrics(logits, labels):
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "recall": recall_score(
            labels, predictions, average="weighted", zero_division=1
        ),
        "precision": precision_score(
            labels, predictions, average="weighted", zero_division=1
        ),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


# Training loop
for epoch in tqdm(range(num_epochs)):
    metrics_sum = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
    num_batches = 0

    for batch in ds["train"].batch(batch_size):
        images = batch["image"].numpy() #, 3, axis=-1)
        labels = batch["label"].numpy()

        def train_step(state, images, labels):
            def loss_fn(params):
                logits, *_ = model1.apply({"params": state.model1_params}, images)
                predictions = model2.apply({"params": params}, logits)
                loss = cross_entropy(predictions, labels)
                return loss, predictions

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, predictions), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss, predictions

        train_state, loss, predictions = train_step(train_state, images, labels)

        metrics_batch = compute_metrics(predictions, labels)
        for key in metrics_sum:
            metrics_sum[key] += metrics_batch[key]
        num_batches += 1

    # Calculate average metrics for the epoch
    for key in metrics_sum:
        metrics_sum[key] /= num_batches

    print(f"Epoch {epoch+1} Metrics:")
    for metric, value in metrics_sum.items():
        print(f"{metric.capitalize()}: {value:.4f}")


# Save the weights
with open("model1_weights.pkl", "wb") as f:
    pickle.dump(train_state.model1_params, f)

with open("model2_weights.pkl", "wb") as f:
    pickle.dump(train_state.params, f)

# save metrics dictionary to a csv
import csv
with open('metrics.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in metrics_sum.items():
       writer.writerow([key, value])

