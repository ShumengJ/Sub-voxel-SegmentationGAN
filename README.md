# Sub-voxel Segmentation GAN

Generative Adversarial Networks trained on 3D micro-CT scans of avian eggshells for sub-voxel pore identification and segmentation.

## Requirements
The model training was conducted on Digital Research Alliance of Canada ([DRAC](https://docs.alliancecan.ca/wiki/Technical_documentation)) compute nodes equipped with NVIDIA V100 GPUs (16 GB). 
To install the requirements, run:
```
pip install -r requirements.txt
```
Then set up the environment:
```
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 cudnn/8.6 opencv/4.5.5
module load python/3.9.6
source $HOME/ENV/bin/activate
```

## Dataset
The dataset directory is organized as follows:
```
/DATASETNAME
  /Train
    /low_res
      /low_res_0001.tif
      /low_res_0002.tif
      ...
      /low_res_0422.tif
    /high_res
      /high_res_0001.tif
      ...
      /high_res_0422.tif
    /segmentation
      /segmentation_0001.tif
      ...
      /segmentation_0422.tif
  /Test
    /low_res
      /low_res_0423.tif
      ...
    /high_res
      /high_res_0423.tif
      ...
    /segmentation
      /segmentation_0423.tif
      ...
```
If you need to change any file paths, adjust the settings in ```global_path.py```, and if dataset organization differs, update ```load_data.py``` accordingly.

## Train or Validate
```
python $HOME/<YOURPATHtoPROJECT>/train.py
```
```
python $HOME/<YOURPATHtoPROJECT>/validation.py
```

## (Optional) Monitoring 
Monitor the training process by launching TensorBoard. 
```
tensorboard --logdir=$HOME/<YOURPATHtoPROJECT>/logs --host 0.0.0.0 --load_fast false &
```
Losses and metrics are also printed to the terminal after each training epoch.

