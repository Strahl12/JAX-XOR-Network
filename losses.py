import jax.numpy as jnp
import optax


def calculate_loss_acc(state, params, batch):
    data_input, labels = batch

    # Obtain logits and model prediction from input data
    logits = state.apply_fn(params, data_input).squeeze(axis=1)
    pred_labels = (logits > 0).astype(jnp.float32)

    # Calculate the loss & accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()
    return loss, acc
