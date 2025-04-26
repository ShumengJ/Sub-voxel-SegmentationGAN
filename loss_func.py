import tensorflow as tf
from global_var import LAMBDA_MSE, LAMBDA_BCE


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def dice_coefficient_per_class(y_true, y_pred, epsilon=1e-6):
    """
    Compute DICE coefficient for each class separately for 3D volumes.
    Input shape: [batch, channels, depth, height, width]
    Returns: tensor of shape [num_classes]
    """
    y_true = tf.cast(y_true, tf.float32)
    # Binarize each channel thresholding prediction at 0.5
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    # Sum over batch and spatial dimensions: depth, height, width.
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 2, 3, 4])
    union = tf.reduce_sum(y_true, axis=[0, 2, 3, 4]) + tf.reduce_sum(y_pred, axis=[0, 2, 3, 4])
    
    dice = (2. * intersection + epsilon) / (union + epsilon)  # shape: [num_classes]
    return dice

def get_class2_metrics(target, prediction):
    """
    Compute TP, TN, FP, FN for class 2 (index = 2) for 3D volumes.
    Inputs: both shape (batch, 3, depth, height, width), with softmax output and one-hot labels.
    """
    # Get predicted and true classes by taking argmax over the channel axis.
    pred_mask = tf.argmax(prediction, axis=1)  # shape: (batch, depth, height, width)
    true_mask = tf.argmax(target, axis=1)
    
    # Create binary masks for Pores (Class 2)
    pred_binary = tf.cast(pred_mask == 2, tf.int32)
    true_binary = tf.cast(true_mask == 2, tf.int32)
    
    # Compute true positives, true negatives, false positives, false negatives over all voxels
    TP = tf.reduce_sum(tf.cast((pred_binary == 1) & (true_binary == 1), tf.int32))
    TN = tf.reduce_sum(tf.cast((pred_binary == 0) & (true_binary == 0), tf.int32))
    FP = tf.reduce_sum(tf.cast((pred_binary == 1) & (true_binary == 0), tf.int32))
    FN = tf.reduce_sum(tf.cast((pred_binary == 0) & (true_binary == 1), tf.int32))
    
    return TP, TN, FP, FN

def generator_loss(disc_generated_output, gen_output, target, input_image):
    
    """
    Compute the total generator loss for a 3D GAN.
    
    Inputs:
      - disc_generated_output: discriminator output for generated samples.
      - gen_output: generated segmentation output,
                    shape (batch, 3, depth, height, width) with softmax probabilities.
      - target: ground truth segmentation mask (one-hot),
                shape (batch, 3, depth, height, width).
      - input_image: input image used by the generator, assumed to be
                     shape (batch, 3, depth, height, width) for a 3-channel input.
    
    The loss comprises:
      - GAN loss (Binary Crossentropy against ones)
      - A residual MSE loss computed between the input image modulated by the predictions and target.
      - A segmentation BCE loss.
    
    Returns:
      total_gen_loss, gan_loss, mse_loss, bce_loss, dice_per_class
    """

    # GAN loss: encourage discriminator to classify generated outputs as real.
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Residual loss: MSE between residual representations.
    # Element-wise multiply input_image with the output and ground truth.
    residual_gen = input_image * gen_output
    residual_gt = input_image * target
    mse_loss = tf.reduce_mean(tf.square(residual_gen - residual_gt))

    # Segmentation BCE loss
    bce = tf.keras.losses.BinaryCrossentropy()
    bce_loss = bce(target, gen_output)

    # Total generator loss with custom weighting (adjust weights as needed)
    total_gen_loss = gan_loss + (LAMBDA_MSE * mse_loss) + (LAMBDA_BCE * bce_loss)

    # Compute DICE coefficient per class.
    dice_per_class = dice_coefficient_per_class(target, gen_output)  
    
    return total_gen_loss, gan_loss, mse_loss, bce_loss, dice_per_class

def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Compute the discriminator loss for a 3D GAN.
    
    Inputs:
      - disc_real_output: discriminator output for real samples.
      - disc_generated_output: discriminator output for generated samples.
      
    Uses Binary Crossentropy loss.
    """
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


