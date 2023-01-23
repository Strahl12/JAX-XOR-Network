from model import Classifier
from losses import calculate_loss_acc
import jax
from constants import NUM_EPOCHS, LEARNING_RATE
from tqdm import tqdm
from dataio import XORDataset, numpy_collate
import torch.utils.data as data
from flax.training import train_state
import optax
import jax.numpy as jnp


@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(calculate_loss_acc,
                                 argnums=1,
                                 has_aux=True
                                 )

    # Determine grads for current model, the params and the batch.
    (loss, acc) = grads = grad_fn(state, state.params, batch)

    # Perform params update
    state = state.apply_gradients(grads=grads)

    # Return new state and other values of interest
    return state, loss, acc


@jax.jit
def eval_step(state, batch):
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc


def train_model(state, data_loader, num_epochs=NUM_EPOCHS):
    # Training Loop
    for epoch in tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss, acc = train_step(state, batch)
    return state


if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)

    train_dataset = XORDataset(size=2500, seed=42)
    train_data_loader = data.DataLoader(train_dataset, batch_size=128,
                                        shuffle=True, collate_fn=numpy_collate)

    model = Classifier(num_hidden=8, num_outputs=1)
    inp = jax.random.normal(inp_rng, (8, 2))
    params = model.init(init_rng, inp)
    optimiser = optax.sgd(learning_rate=LEARNING_RATE)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=optimiser)
    trained_model_state = train_model(model_state, train_data_loader,
                                      num_epochs=NUM_EPOCHS)
