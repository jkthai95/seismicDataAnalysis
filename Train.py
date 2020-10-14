import tensorflow as tf
import matplotlib.pyplot as plt
import os

import MyModel
import Dataset


def loss(model, seismic_data, labels, loss_function):
    labels_pred = model(seismic_data, training=True)

    return loss_function(y_true=labels, y_pred=labels_pred)


def grad(model, inputs, targets, loss_function):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, loss_function)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train_model(config):
    # Acquire TFRecord dataset
    train_dataset = Dataset.acquire_tfrecord_dataset('train', config)
    valid_dataset = Dataset.acquire_tfrecord_dataset('valid', config)

    # Define new model
    model = MyModel.MyModel(drop_rate=config.drop_rate)

    # Acquire model
    if os.path.exists(config.model_path):
        # Load existing pre-trained model
        model.load_weights(config.model_path)
    else:
        # Ensure model directory exists
        model_dir = os.path.dirname(config.model_path)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

    # Loss function
    loss_function = tf.keras.losses.MeanSquaredError()

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Keep results for plotting
    train_loss_results = []
    valid_loss_results = []
    best_loss_result = 100

    num_epochs = config.num_epochs
    num_epochs_wait = 0
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()

        # Training loop
        for sample in train_dataset:
            seismic_data = sample["seismic"]
            labels = sample["labels"]

            # Optimize the model
            loss_value, grads = grad(model, seismic_data, labels, loss_function)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss

        # Test on validation data
        valid_loss = tf.metrics.MeanSquaredError()
        for sample in valid_dataset:
            seismic_data = sample["seismic"]
            labels = sample["labels"]

            valid_loss.update_state(labels, model(seismic_data))

        if valid_loss.result() < best_loss_result:
            # Validation loss improved, save model
            best_loss_result = valid_loss.result()
            model.save_weights(config.model_path)
            num_epochs_wait = 0
        else:
            # Increment how much epochs model did not improve
            num_epochs_wait = num_epochs_wait + 1

        # Save results
        train_loss_results.append(epoch_loss_avg.result())
        valid_loss_results.append(valid_loss.result())
        print("Epoch {}: Train loss = {:5f}, Valid loss = {:f}".format(epoch, epoch_loss_avg.result(), valid_loss.result()))

        if num_epochs_wait >= config.patients:
            # Early stopping for training
            break

    # Visualize loss and accuracy
    fig, axes = plt.subplots(2, sharex=True)
    fig.suptitle('MSE Loss')

    axes[0].set_ylabel("Train loss")
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Valid loss")
    axes[1].set_xlabel("Epoch")
    axes[1].plot(valid_loss_results)
    plt.show()

