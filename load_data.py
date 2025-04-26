import tifffile as tiff
import numpy as np
import tensorflow as tf

print('Tensorflow version: ', tf.__version__)



def load_tiff(low_res_file, high_res_file, seg_file):
    """
    Loads separate TIFF files for low-resolution, high-resolution, and segmentation volumes.
    Converts file paths (tf.Tensor) to Python strings before reading.
    
    Assumes each TIFF file is a 3D volume of shape (256, 256, 256).
    """
    # Convert the tf.Tensor file paths to Python strings.
    # Use .numpy() because low_res_file is a tf.Tensor in eager mode.
    lr_path = low_res_file.numpy().decode('utf-8')
    hr_path = high_res_file.numpy().decode('utf-8')
    seg_path = seg_file.numpy().decode('utf-8')
    
    # Load the TIFF files using the converted string paths.
    low_res_volume = tiff.imread(lr_path)
    high_res_volume = tiff.imread(hr_path)
    seg_volume = tiff.imread(seg_path)
    
    return low_res_volume, high_res_volume, seg_volume



def normalize_to_uint8(image):
    """
    Normalize a floating-point image to uint8 range (0–255).
    """
    image = image.numpy()  # Convert TensorFlow tensor to NumPy array
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize to [0, 1]
    image = (image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    return image



