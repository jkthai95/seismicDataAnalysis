import MyModel
import Config
import tensorflow as tf


def loss(model, seismic_data, labels, loss_object):
    labels_pred = model(seismic_data)

    return loss_object(y_true=labels, y_pred=labels_pred)


def grad(model, inputs, targets, loss_object):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, loss_object)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train_model(model, train_dataset, config):
    # Loss function
    loss_object = tf.keras.losses.MeanSquaredError()

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = config.num_epochs

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop
        for sample in train_dataset:
            seismic_data = sample["seismic"]
            labels = sample["labels"]

            # Optimize the model
            loss_value, grads = grad(model, seismic_data, labels, loss_object)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(labels, model(seismic_data))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))



