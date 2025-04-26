import os
import tensorflow as tf
from load_data import load_tiff
from global_var import PATCH_SIZE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define constants
ORIG_SIZE = PATCH_SIZE      # original volume size 
PAD_SIZE = 15               # pad 15 voxels on each side so that padded dims: 256 + 2*15 = 286
NEW_SIZE = ORIG_SIZE        # back to original size 


def random_jitter_3d(lr_vol, seg_vol):

    """
    Apply random 3D spatial jittering and flips to a low-resolution volume and its segmentation.

    Ensures inputs are 4D (adds a channel dimension if needed), pads with reflection,
    takes a random crop of size NEW_SIZE, and randomly flips along each axis.

    Parameters
    ----------
    lr_vol : tf.Tensor
        Low-resolution volume, shape [D, H, W] or [D, H, W, C].
    seg_vol : tf.Tensor
        Segmentation volume corresponding to `lr_vol`, same shape.

    Returns
    -------
    lr_crop : tf.Tensor
        Jittered low-resolution volume, shape [NEW_SIZE, NEW_SIZE, NEW_SIZE, C].
    seg_crop : tf.Tensor
        Jittered segmentation volume, shape [NEW_SIZE, NEW_SIZE, NEW_SIZE, C].

    Notes
    -----
    - PAD_SIZE, ORIG_SIZE, and NEW_SIZE are module-level constants.
    - Padding is applied with mode='REFLECT'.
    - The random crop start indices for each axis are uniformly sampled 
      from [0, padded_dim - NEW_SIZE].
    - Each axis (depth, height, width) is flipped independently with 50% probability.
    """

    # Ensure the volumes have rank 4. If they have rank 3, add a channel dimension.
    if lr_vol.shape.ndims == 3:
        lr_vol = tf.expand_dims(lr_vol, axis=-1)
    if seg_vol.shape.ndims == 3:
        seg_vol = tf.expand_dims(seg_vol, axis=-1)
    
    # Define paddings for a rank-4 tensor: pad depth, height, width, and no pad for channels.
    paddings = [[PAD_SIZE, PAD_SIZE], [PAD_SIZE, PAD_SIZE], [PAD_SIZE, PAD_SIZE], [0, 0]]
    
    # Apply reflect padding to both volumes.
    lr_padded  = tf.pad(lr_vol, paddings, mode='REFLECT')
    seg_padded = tf.pad(seg_vol, paddings, mode='REFLECT')
    
    # Determine padded spatial dimension (should be 286).
    padded_dim = ORIG_SIZE + 2 * PAD_SIZE 
    
    # Randomly select starting indices for depth, height, and width.
    d_start = tf.random.uniform((), minval=0, maxval=padded_dim - NEW_SIZE + 1, dtype=tf.int32)
    h_start = tf.random.uniform((), minval=0, maxval=padded_dim - NEW_SIZE + 1, dtype=tf.int32)
    w_start = tf.random.uniform((), minval=0, maxval=padded_dim - NEW_SIZE + 1, dtype=tf.int32)
    
    # Crop the padded volumes back to the original size.
    lr_crop = lr_padded[d_start:d_start+NEW_SIZE,
                        h_start:h_start+NEW_SIZE,
                        w_start:w_start+NEW_SIZE, :]
    seg_crop = seg_padded[d_start:d_start+NEW_SIZE,
                          h_start:h_start+NEW_SIZE,
                          w_start:w_start+NEW_SIZE, :]
    
    # Random flipping along each spatial axis.
    if tf.random.uniform(()) > 0.5:
        lr_crop = tf.reverse(lr_crop, axis=[0])
        seg_crop = tf.reverse(seg_crop, axis=[0])
    if tf.random.uniform(()) > 0.5:
        lr_crop = tf.reverse(lr_crop, axis=[1])
        seg_crop = tf.reverse(seg_crop, axis=[1])
    if tf.random.uniform(()) > 0.5:
        lr_crop = tf.reverse(lr_crop, axis=[2])
        seg_crop = tf.reverse(seg_crop, axis=[2])
    
    return lr_crop, seg_crop



def normalize_3d(lr_vol):

    """
    Scale a 3D volume tensor to the range [-1, 1] based on its maximum value.

    Parameters
    ----------
    lr_vol : tf.Tensor
        Input volume of shape [D, H, W] or [D, H, W, C], with arbitrary value range.

    Returns
    -------
    lr_norm : tf.Tensor
        Normalized volume of the same shape as `lr_vol`, scaled so that the
        maximum value maps to +1 and the minimum stays ≥ -1.
        If the input’s maximum is zero, returns the original tensor unchanged.

    Notes
    -----
    - Uses `tf.reduce_max` to find the maximum voxel intensity.
    - Guards against division by zero by checking if the max is 0 before scaling.
    - The operation `(lr_vol / max_val) * 2 - 1` maps [0, max_val] → [-1, +1].
    """

    max_val = tf.reduce_max(lr_vol)

    # Avoid division by zero if max_val happens to be 0
    lr_norm = tf.cond(tf.equal(max_val, 0), lambda: lr_vol, lambda: (lr_vol / max_val) * 2 - 1)
    return lr_norm





@tf.function(reduce_retracing=True)
def random_jitter(lr_vol, seg_vol):
    return random_jitter_3d(lr_vol, seg_vol)





def load_tiff_train(low_res_file, high_res_file, seg_file):

    """
    Load paired low- and high-resolution TIFF volumes and segmentation, apply augmentation,
    normalize, and convert the segmentation to one-hot encoding for training.

    Parameters
    ----------
    low_res_file : str or Path
        Path to the low-resolution TIFF file.
    high_res_file : str or Path
        Path to the high-resolution TIFF file.
    seg_file : str or Path
        Path to the segmentation TIFF file (labels in {1, 2, 3}).

    Returns
    -------
    lr_aug : tf.Tensor
        Augmented and normalized low-resolution volume, shape [NEW_SIZE, NEW_SIZE, NEW_SIZE, 1],
        with values in [-1, 1].
    seg_onehot : tf.Tensor
        One-hot encoded segmentation tensor, shape [NEW_SIZE, NEW_SIZE, NEW_SIZE, 3],
        where the original labels {1, 2, 3} map to indices {0, 1, 2}.

    Notes
    -----
    - Assumes `load_tiff` returns `(lr, hr, seg)` each as a 4D tensor of shape (256, 256, 256, 1).
    - Applies `random_jitter` to `(lr, seg)` for spatial augmentations.
    - Uses `normalize_3d` to scale the low-resolution volume to [-1, 1].
    - Converts the integer segmentation labels to one-hot via `tf.one_hot`.
    """

    # Load volumes (assumes load_tiff_files returns tensors with shape (256,256,256,1))
    lr, hr, seg = load_tiff(low_res_file, high_res_file, seg_file)
    
    # Apply random jitter augmentation to low-res and segmentation volumes.
    lr_aug, seg_aug = random_jitter(lr, seg)
    
    # Normalize the low-res volume.
    lr_aug = normalize_3d(lr_aug)
    
    # Convert segmentation from shape (256,256,256,1) with values {1,2,3} to one-hot encoded tensor.
    # First squeeze the channel dimension: (256,256,256) of type int32.
    seg_int = tf.cast(tf.squeeze(seg_aug, axis=-1), tf.int32)

    # One-hot encode. (Subtract 1 so that label 1 becomes index 0, etc.)
    seg_onehot = tf.one_hot(seg_int - 1, depth=3, axis=-1)
    
    return lr_aug, seg_onehot




def load_tiff_val(low_res_file, high_res_file, seg_file):

    lr, hr, seg = load_tiff(low_res_file, high_res_file, seg_file)
    lr_norm = normalize_3d(lr)
    
    # Ensure that lr_norm has a channel dimension.
    # If lr_norm is rank 3 ([256,256,256]), add a new axis at the end.
    if tf.rank(lr_norm) == 3:
        lr_norm = tf.expand_dims(lr_norm, axis=-1)
    
    # Process the segmentation volume.
    # If seg has an extra channel (size 1), squeeze it; else leave it.
    seg_int = tf.cond(
        tf.equal(tf.shape(seg)[-1], 1), lambda: tf.cast(tf.squeeze(seg, axis=-1), tf.int32), lambda: tf.cast(seg, tf.int32)
    )
    # Convert to one-hot encoding (maps labels {1,2,3} to one-hot channels).
    seg_onehot = tf.one_hot(seg_int - 1, depth=3, axis=-1)
    
    return lr_norm, seg_onehot