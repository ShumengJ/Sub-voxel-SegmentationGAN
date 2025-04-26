import tensorflow as tf
from global_var import OUTPUT_CHANNELS, INPUT_CHANNELS



############################################
# 3D Downsample and Upsample helper layers #
############################################

def downsample(filters, size, apply_batchnorm=True):
    """
    3D downsampling block using Conv3D (strides=2 reduces each spatial dimension by half).
    Uses channels_first data format.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    # Use a cubic kernel: size is applied to depth, height and width.
    result.add(tf.keras.layers.Conv3D(filters, kernel_size=size, strides=2,
                                      padding='same', kernel_initializer=initializer,
                                      use_bias=False, data_format='channels_first'))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization(axis=1))
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    """
    3D upsampling block using Conv3DTranspose (strides=2 doubles each spatial dimension).
    Uses channels_first data format.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv3DTranspose(filters, kernel_size=size, strides=2,
                                               padding='same', kernel_initializer=initializer,
                                               use_bias=False, data_format='channels_first'))
    result.add(tf.keras.layers.BatchNormalization(axis=1))
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

############################################
# 3D Generator Model                       #
############################################

def Generator():
    """
    3D Generator with skip connections. Expects an input volume of shape 
    [INPUT_CHANNELS, 256, 256, 256] and returns an output volume of shape 
    [OUTPUT_CHANNELS, 256, 256, 256] with a softmax applied across the channel axis.
    """
    inputs = tf.keras.layers.Input(shape=[INPUT_CHANNELS, 256, 256, 256])
    
    # Build the downsampling stack; each downsample layer halves the spatial dimensions.
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),   # (batch, 64, 128, 128, 128)
        downsample(128, 4),                         # (batch, 128, 64, 64, 64)
        downsample(256, 4),                         # (batch, 256, 32, 32, 32)
        downsample(512, 4),                         # (batch, 512, 16, 16, 16)
        downsample(512, 4),                         # (batch, 512, 8, 8, 8)
        downsample(512, 4),                         # (batch, 512, 4, 4, 4)
        downsample(512, 4),                         # (batch, 512, 2, 2, 2)
        downsample(512, 4),                         # (batch, 512, 1, 1, 1)
    ]

    # Build the upsampling stack; each upsample layer doubles the spatial dimensions.
    up_stack = [
        upsample(512, 4, apply_dropout=True),       # (batch, 512, 2, 2, 2)
        upsample(512, 4, apply_dropout=True),       # (batch, 512, 4, 4, 4)
        upsample(512, 4, apply_dropout=True),       # (batch, 512, 8, 8, 8)
        upsample(512, 4),                           # (batch, 512, 16, 16, 16)
        upsample(256, 4),                           # (batch, 256, 32, 32, 32)
        upsample(128, 4),                           # (batch, 128, 64, 64, 64)
        upsample(64, 4),                            # (batch, 64, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    # Final layer to restore spatial dimensions to 256 in all three dimensions.
    last = tf.keras.layers.Conv3DTranspose(OUTPUT_CHANNELS, kernel_size=4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             activation=None,  # (use softmax later)
                                             data_format='channels_first')  # (batch, OUTPUT_CHANNELS, 256, 256, 256)
    
    x = inputs
    skips = []

    # Downsampling – store skip connections.
    for down in down_stack:
        x = down(x)
        skips.append(x)
        
    # Reverse the order of skip connections, except the last one.
    skips = list(reversed(skips[:-1]))
    
    # Upsampling with skip connections.
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate(axis=1)([x, skip])
    x = last(x)
    
    # Apply softmax over the channels (axis 1) to yield one-hot outputs.
    x_soft = tf.keras.layers.Softmax(axis=1, name="softmax_output")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x_soft)


############################################
# 3D Discriminator Model                   #
############################################

def Discriminator():
    """
    3D PatchGAN Discriminator. Expects two inputs:
      - input_image: shape [INPUT_CHANNELS, 256, 256, 256]
      - target_image: shape [OUTPUT_CHANNELS, 256, 256, 256]
    These are concatenated and passed through a series of 3D convolutions.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = tf.keras.layers.Input(shape=[INPUT_CHANNELS, 256, 256, 256], name='input_image')
    tar = tf.keras.layers.Input(shape=[OUTPUT_CHANNELS, 256, 256, 256], name='target_image')
    
    # Concatenate along channels (axis=1)
    x = tf.keras.layers.Concatenate(axis=1)([inp, tar])    # shape: [batch, INPUT_CHANNELS+OUTPUT_CHANNELS, 256, 256, 256]
    
    # Apply successive downsampling blocks.
    down1 = downsample(64, 4, apply_batchnorm=False)(x)    # (batch, 64, 128, 128, 128)
    down2 = downsample(128, 4)(down1)                      # (batch, 128, 64, 64, 64)
    down3 = downsample(256, 4)(down2)                      # (batch, 256, 32, 32, 32)
    
    # Zero-padding with 3D padding. Here we pad one voxel in each dimension.
    zero_pad1 = tf.keras.layers.ZeroPadding3D(padding=((1,1), (1,1), (1,1)), data_format='channels_first')(down3)
    
    conv = tf.keras.layers.Conv3D(512, kernel_size=4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False,
                                  data_format='channels_first')(zero_pad1)
    
    batchnorm1 = tf.keras.layers.BatchNormalization(axis=1)(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    
    zero_pad2 = tf.keras.layers.ZeroPadding3D(padding=((1,1), (1,1), (1,1)), data_format='channels_first')(leaky_relu)
    
    last = tf.keras.layers.Conv3D(1, kernel_size=4, strides=1,
                                  kernel_initializer=initializer,
                                  data_format='channels_first')(zero_pad2) 
    
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
