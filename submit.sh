#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=190000M        # memory per node
#SBATCH --time=3-00:00      # time (DD-HH:MM)

module load StdEnv/2020 gcc/9.3.0 cuda/11.8 cudnn/8.6 opencv/4.5.5
module load python/3.9.6
source $HOME/ENV/bin/activate

tensorboard --logdir=$HOME/scratch/Finch_Aug2024/Tensorflow11/seg_dis_slow/logs --host 0.0.0.0 --load_fast false &
python $HOME/scratch/Finch_Aug2024/Tensorflow11/seg_dis_slow/train.py