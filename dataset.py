import os
import tensorflow as tf
from pathlib import Path

from global_var import BATCH_SIZE, PATCH_SIZE
from global_path import HOMEPATH, DATAPATH
from data_aug import load_tiff_train, load_tiff_val


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# Define ROOTPATH and folder structure
train_base = Path(HOMEPATH + DATAPATH)  / "Train"
test_base  = Path(HOMEPATH + DATAPATH)  / "Test"

# Directories for train
train_lr_dir = str(train_base / "low_res")
train_hr_dir = str(train_base / "high_res")
train_seg_dir = str(train_base / "segmentation")

train_num = int(len(os.listdir(train_lr_dir))) 
print('Number of training patches: ', train_num)

# Directories for test
test_lr_dir = str(test_base / "low_res")
test_hr_dir = str(test_base / "high_res")
test_seg_dir = str(test_base / "segmentation")

test_num = int(len(os.listdir(test_lr_dir))) 
print('Number of testing patches:  ', test_num)


def load_patch_train(low_res_path, high_res_path, seg_path):

    """
    Wrap `load_tiff_train` for use in a `tf.data` pipeline, extracting fixed-size patches.

    Calls `load_tiff_train` via `tf.py_function` to load and preprocess volumes,
    then sets the static shape to match `PATCH_SIZE`.

    Parameters
    ----------
    low_res_path : tf.Tensor or str
        Scalar string tensor or filepath to the low-resolution TIFF.
    high_res_path : tf.Tensor or str
        Scalar string tensor or filepath to the high-resolution TIFF.
    seg_path : tf.Tensor or str
        Scalar string tensor or filepath to the segmentation TIFF.

    Returns
    -------
    lr : tf.Tensor
        Preprocessed low-resolution patch, shape [PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1],
        dtype tf.float32, values in [-1, 1].
    seg : tf.Tensor
        One-hot encoded segmentation patch, shape [PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 3],
        dtype tf.float32.

    Notes
    -----
    - Uses `tf.py_function` to wrap the Python-side `load_tiff_train` function.
    - `set_shape` is called to inform TensorFlow of the static output dimensions.
    - Intended for mapping over file paths in a `tf.data.Dataset`.
    """
        
    lr, seg = tf.py_function(load_tiff_train, [low_res_path, high_res_path, seg_path], [tf.float32, tf.float32])
    lr.set_shape([PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1])
    seg.set_shape([PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 3])
    return lr, seg

def load_patch_val(low_res_path, high_res_path, seg_path):

    lr, seg = tf.py_function(load_tiff_val, [low_res_path, high_res_path, seg_path], [tf.float32, tf.float32])
    lr.set_shape([PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1])
    seg.set_shape([PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 3])
    return lr, seg




def create_dataset_train(lr_dir, hr_dir, seg_dir, batch_size, shuffle=True, shuffle_buffer=100):

    """
    Create a TensorFlow training dataset of paired low-res, high-res, and segmentation patches.

    Lists TIFF file paths in the specified directories, zips them into triplets,
    optionally shuffles, applies preprocessing to load and augment patches,
    and batches the results.

    Parameters
    ----------
    lr_dir : str or Path
        Directory containing low-resolution TIFF files named like "low_res_*.tif".
    hr_dir : str or Path
        Directory containing high-resolution TIFF files named like "high_res_*.tif".
    seg_dir : str or Path
        Directory containing segmentation TIFF files named like "hr_seg_*.tif".
    batch_size : int
        Number of samples per batch.
    shuffle : bool, optional
        Whether to shuffle the file paths before loading (default: True).
    shuffle_buffer : int, optional
        Buffer size for shuffling (default: 100).

    Returns
    -------
    tf.data.Dataset
        A `Dataset` yielding batches of `(lr_patch, seg_onehot)` tuples:
        - `lr_patch`: tf.Tensor of shape `[batch_size, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1]`, dtype `tf.float32`.
        - `seg_onehot`: tf.Tensor of shape `[batch_size, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 3]`, dtype `tf.float32`.
    """

    # List file names
    lr_files = tf.data.Dataset.list_files(os.path.join(lr_dir, "low_res_*.tif"), shuffle=False)
    hr_files = tf.data.Dataset.list_files(os.path.join(hr_dir, "high_res_*.tif"), shuffle=False)
    seg_files = tf.data.Dataset.list_files(os.path.join(seg_dir, "hr_seg_*.tif"), shuffle=False)
    
    # Zip the datasets so each element is a triple: (lr_path, hr_path, seg_path)
    dataset = tf.data.Dataset.zip((lr_files, hr_files, seg_files))
    
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    
    # Map load_patch_train which returns (lr_patch, seg_onehot)
    dataset = dataset.map(load_patch_train, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    return dataset



def create_dataset_val(lr_dir, hr_dir, seg_dir, batch_size, shuffle=False, shuffle_buffer=100):

    # List file names
    lr_files = tf.data.Dataset.list_files(os.path.join(lr_dir, "low_res_*.tif"), shuffle=False)
    hr_files = tf.data.Dataset.list_files(os.path.join(hr_dir, "high_res_*.tif"), shuffle=False)
    seg_files = tf.data.Dataset.list_files(os.path.join(seg_dir, "hr_seg_*.tif"), shuffle=False)
    
    # Zip the datasets
    dataset = tf.data.Dataset.zip((lr_files, hr_files, seg_files))
    
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    
    # Map load_patch_val which returns (lr_patch, seg_onehot)
    dataset = dataset.map(load_patch_val, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    return dataset


# Create training dataset
train_dataset = create_dataset_train(train_lr_dir, train_hr_dir, train_seg_dir, batch_size=BATCH_SIZE, shuffle=True)

# Create testing dataset (no shuffling)
test_dataset = create_dataset_val(test_lr_dir, test_hr_dir, test_seg_dir, batch_size=BATCH_SIZE, shuffle=False)








