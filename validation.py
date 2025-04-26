import tensorflow as tf
import numpy as np

from train import generate_images, generator, checkpoint, checkpoint_dir
from dataset import test_dataset


# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

test_dataset = test_dataset.prefetch(1)    # never buffer more than one batch

id = 000
sum_tp = np.zeros(3, int)
sum_tn = np.zeros(3, int)
sum_fp = np.zeros(3, int)
sum_fn = np.zeros(3, int)

for id, (inp, tar) in enumerate(test_dataset, start=0):

    # NHWC→NCHW
    inp = tf.transpose(inp, [0,4,1,2,3])
    tar = tf.transpose(tar, [0,4,1,2,3])

    # write + get metrics
    tp_list, tn_list, fp_list, fn_list = generate_images(generator, inp, tar, id, val=True)

    # accumulate
    sum_tp += np.array(tp_list)
    sum_tn += np.array(tn_list)
    sum_fp += np.array(fp_list)
    sum_fn += np.array(fn_list)




# --- after the loop: compute metrics ---
eps = 1e-8
total = sum_tp + sum_tn + sum_fp + sum_fn  # per class

accuracy_per    = (sum_tp + sum_tn) / (total + eps)
precision_per   = sum_tp / (sum_tp + sum_fp + eps)
recall_per      = sum_tp / (sum_tp + sum_fn + eps)
specificity_per = sum_tn / (sum_tn + sum_fp + eps)
dice_per          = 2*sum_tp / (2*sum_tp + sum_fp + sum_fn + eps)

# macro‑averages
accuracy_macro    = accuracy_per.mean()
precision_macro   = precision_per.mean()
recall_macro      = recall_per.mean()
specificity_macro = specificity_per.mean()
dice_macro        = dice_per.mean()


# --- print results with separators ---
print("----")
for c in range(3):
    print(
        f"| Class {c+1} | "
        f"Acc {accuracy_per[c]:.4f} | "
        f"Prec {precision_per[c]:.4f} | "
        f"Rec {recall_per[c]:.4f} | "
        f"Spec {specificity_per[c]:.4f} | "
        f"Dice {dice_per[c]:.4f} |"
    )
print("----")
print(
    f"| Macro   | "
    f"Acc {accuracy_macro:.4f} | "
    f"Prec {precision_macro:.4f} | "
    f"Rec {recall_macro:.4f} | "
    f"Spec {specificity_macro:.4f} | "
    f"Dice {dice_macro:.4f} |"
)
print("----")