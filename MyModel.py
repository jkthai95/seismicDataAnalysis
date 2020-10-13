import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import activations

def crop_and_concat(x1, x2):
    """
    Crops x1 tensor and concatenates it to x2
    :param x1: input tensor 1
    :param x2: input tensor 2
    :return: Concatenated tensor
    """
    x1_shape = x1.shape
    x2_shape = x2.shape

    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3]]
    x1_crop = tf.slice(x1, offsets, size)
    return layers.Concatenate()([x1_crop, x2])


def conv_block_down(inputs, num_filters, stride=2, drop_rate=None):
    x = layers.Conv2D(num_filters, 3, strides=(stride, stride), padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    if drop_rate and (drop_rate > 0):
        x = layers.Dropout(drop_rate)(x)

    return x


def conv_block_up(inputs, num_filters, stride=2, drop_rate=None):
    x = layers.Conv2DTranspose(num_filters, 3, strides=(stride, stride), padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    if drop_rate and (drop_rate > 0):
        x = layers.Dropout(drop_rate)(x)

    return x


class MyModel:
    def __init__(self, drop_rate=None):
        super(MyModel, self).__init__()

        # Parameters
        self.drop_rate = drop_rate

    def get_model(self, inputs):
        # Build stem
        x = layers.Conv2D(32, 3, strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activations.relu)(x)

        # Downsampling layers
        down1 = conv_block_down(x, 64, 2, self.drop_rate)
        down2 = conv_block_down(down1, 128, 2, self.drop_rate)

        # Bottleneck layer
        bottleneck = conv_block_down(down2, 256, 1, self.drop_rate)
        bottleneck = conv_block_up(bottleneck, 128, 2, self.drop_rate)

        # Upsampling layers
        up2 = crop_and_concat(bottleneck, down2)
        up2 = conv_block_up(up2, 64, 2, self.drop_rate)
        up1 = crop_and_concat(up2, down1)
        up1 = conv_block_up(up1, 64, 2, self.drop_rate)

        # Build top
        x = crop_and_concat(up1, x)
        x = layers.Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.Activation(activations.relu)(x)
        if self.drop_rate and (self.drop_rate > 0):
            x = layers.Dropout(self.drop_rate)(x)
        outputs = layers.Conv2D(1, 3, padding='same', activation=activations.tanh)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model
