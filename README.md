# Sub-voxel Segmentation GAN

This repository accompanies our paper, Jia *et al.* (2025), on multiscale X-ray
microcomputed tomography (µCT) of avian eggshells. It provides the released
TensorFlow implementation of a model developed to infer voxel-wise background,
eggshell, and pore labels from lower-resolution 3D data, including pore features
below the input voxel scale.

## Scientific context

The study addresses the field-of-view/resolution trade-off in eggshell µCT.
Registered low- and high-resolution scans of ostrich, guillemot, and crow
eggshells are used to train a 3D network that combines resolution enhancement
with segmentation. The framework emphasizes 3D context, 3D convolution, class
balancing, and edge attention because pore voxels are exceptionally sparse.

The implementation uses a channels-first 3D conditional GAN:

- a U-Net-like generator with eight downsampling levels, skip connections, and
  a three-class softmax output;
- a 3D PatchGAN discriminator conditioned on the low-resolution volume;
- adversarial binary cross-entropy, the edge-attentive residual MSE described
  in the paper, and segmentation binary cross-entropy losses; and
- Dice plus pore-class confusion metrics.

## Repository map

```text
config/example.json               Example run configuration
requirements-legacy.txt           Historical Compute Canada environment snapshot
scripts/train.py                   Source-checkout training entry point
scripts/evaluate.py                Checkpoint inference and evaluation
scripts/plot_models.py             Optional architecture diagrams
src/subvoxel_segmentationgan/
  augmentation.py                 Paired 3D jitter and preprocessing
  data.py                         File matching and tf.data construction
  models.py                       Generator and discriminator
  losses.py                       Released losses and voxel metrics
  training.py                     Training loop, logs, checkpoints
  evaluation.py                   Test inference and aggregate metrics
tests/                             Lightweight path/config/metric tests
```

## Environment setup

The original experiments were run with Python 3.9.6, TensorFlow 2.11, CUDA
11.8, cuDNN 8.6, and an NVIDIA V100 GPU with 16 GB memory. TensorFlow 2.11
supports Python 3.9 and 3.10; use one of those versions.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development checks, install `-e '.[dev]'`. Model diagram generation also
requires `-e '.[plot]'` and a system Graphviz installation. The 256³
channels-first workflow is intended for a GPU-capable TensorFlow environment.

`requirements-legacy.txt` preserves the original Compute Canada environment
snapshot for historical reference. It is not the recommended installation
route; use `requirements.txt` or `pyproject.toml` for current installations.

## Data expectations

The default relative workspace restores the original data organization:

```text
OstrichR496/
  R496_3d/
    Train/
      low_res/low_res_0001.tif
      high_res/high_res_0001.tif
      segmentation/hr_seg_0001.tif
    Test/
      low_res/low_res_0423.tif
      high_res/high_res_0423.tif
      segmentation/hr_seg_0423.tif
```

Each split must contain one or more triplets. Discovery fails early if a
suffix is missing or duplicated. Segmentation files may use the executable
`hr_seg_<suffix>.tif` name or the README-era
`segmentation_<suffix>.tif` alias, but not both for the same suffix. The
implementation expects cubic 256 × 256 × 256 volumes, one low-resolution input
channel, and integer labels `1`, `2`, and `3`, mapped respectively to the three
output channels. Training uses reflection padding, a random translated crop,
independent axis flips, and normalization of low-resolution intensities to
`[-1, 1]`.

Pass either the `R496_3d` directory or its immediate parent to `--data-dir`.
The loader intentionally reads direct modality children and does not guess
deeper specimen/sample layouts.

## Configuration and paths

All locations are relative or configurable; no user home or cluster path is
embedded in the code. Copy `config/example.json` outside the repository if
desired. Relative paths in a JSON configuration are resolved relative to that
file. Command-line arguments override configuration values.

```bash
python scripts/train.py --help
python scripts/evaluate.py --help
```

Built-in path defaults are:

```text
data:        OstrichR496/R496_3d
run root:    OstrichR496/Tensorflow/SRSegGAN
checkpoints: <run root>/training_checkpoints
TensorBoard: <run root>/logs/fit/<timestamp>
epoch log:   <run root>/epoch_loss_log.txt
predictions: <run root>/output
```

Use `--output-dir` to replace the run root. `--checkpoint-dir`, `--log-dir`,
and `--prediction-dir` override individual locations. Other defaults remain
batch size 1, patch size 256, 300 epochs, and the released loss weights. The
eight-level U-Net requires a patch size divisible by 256.

## Training

After installing the package and preparing data, the expected invocation is:

```bash
python scripts/train.py --config config/example.json
```

Equivalent installed entry point:

```bash
subvoxel-train
```

Explicit relative roots remain supported:

```bash
subvoxel-train \
  --data-dir datasets/R496_3d \
  --output-dir runs/experiment-01
```

Training writes `config.json`, `epoch_loss_log.txt`, timestamped TensorBoard
event files under `logs/fit/`, and TensorFlow checkpoints under
`training_checkpoints/` unless an individual path override is supplied.
Monitor the default local run with:

```bash
tensorboard --logdir OstrichR496/Tensorflow/SRSegGAN/logs
```

## Inference and evaluation

Evaluate the `Test` split using an existing checkpoint:

```bash
python scripts/evaluate.py \
  --data-dir datasets/R496_3d \
  --checkpoint-dir runs/experiment-01/training_checkpoints \
  --prediction-dir runs/experiment-01/output
```

The evaluator uses the latest checkpoint and invokes the generator with
`training=True`, matching the released evaluation workflow. By default, it
writes full volumes under
`output/GAN_TIF_val/{low_res,target,output}/`, first-slice previews under
`output/GAN_PNG_val/{low_res,target,output}/`, and aggregate `metrics.csv` and
`metrics.json` under `output/`. The output helper also retains the corresponding
`GAN_TIF` and `GAN_PNG` names for non-validation predictions.

Metrics are reported per class and as unweighted macro averages: accuracy,
precision, recall, specificity, and Dice. Class 3 is the pore class in the
labeling convention used in the study. Interpret accuracy cautiously under the
severe class imbalance described in the paper.

## Software checks

Checks that do not require the dataset or a GPU:

```bash
python -m compileall src scripts tests
pytest
python scripts/train.py --help
python scripts/evaluate.py --help
```

These checks cover packaging, configuration, file pairing, metrics, and CLI
parsing.

## Citation

Please cite the article:

> Jia, S., Piché, N., McKee, M. D., & Reznikov, N. (2025). Advancing X-ray
> microcomputed tomography image processing of avian eggshells: An improved
> registration metric for multiscale 3D images and resolution-enhanced
> segmentation of eggshell pores using edge-attentive neural networks.
> *Micron, 199*, 103915.
> <https://doi.org/10.1016/j.micron.2025.103915>

```bibtex
@article{jia2025avian,
  title   = {Advancing X-ray microcomputed tomography image processing of
             avian eggshells: An improved registration metric for multiscale
             3D images and resolution-enhanced segmentation of eggshell pores
             using edge-attentive neural networks},
  author  = {Jia, Shumeng and Pich{\'e}, Nicolas and McKee, Marc D. and
             Reznikov, Natalie},
  journal = {Micron},
  volume  = {199},
  pages   = {103915},
  year    = {2025},
  doi     = {10.1016/j.micron.2025.103915}
}
```

## Licence

Repository code is released under the MIT License; see `LICENSE`. The article,
datasets, and any third-party images are governed by their own terms.
