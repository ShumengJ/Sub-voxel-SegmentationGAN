import os
import datetime
import cv2
import imageio
import tifffile
import sys
import time
import numpy as np
import tensorflow as tf

from IPython import display
from dataset import train_dataset, test_dataset, train_num
from global_path import HOMEPATH, OUTPUTPATH, FOLDERPATH
from global_var import EPOCH, OUTPUT_CHANNELS
from model_gan import Generator, Discriminator
from loss_func import generator_loss, discriminator_loss, get_class2_metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
print('System version: ', sys.version)
print(os.path.abspath(__file__))

# Set random seeds
tf.random.set_seed(1)
np.random.seed(1)

OUTPUTPATH_tif = HOMEPATH + OUTPUTPATH + '/GAN_TIF'
OUTPUTPATH_png = HOMEPATH + OUTPUTPATH + '/GAN_PNG'





def convert_to_uint8(slice_img):
    """
    Normalize the input 2D slice to 0-255 and convert to uint8.
    """
    # If the image has little variation, avoid division by zero.
    min_val = slice_img.min()
    max_val = slice_img.max()
    if max_val > min_val:
        norm = (slice_img - min_val) / (max_val - min_val)
    else:
        norm = slice_img - min_val
    return (norm * 255).astype(np.uint8)





def generate_images(model, test_input, tar, id, val=False):

    # Run inference: model output shape: [batch, 3, depth, height, width]
    prediction = model(test_input, training=True)
    
    # Map prediction to a one-channel output.
    # Compute argmax along channel axis (axis=1), then convert back using a weighted sum.
    # shape: [batch, depth, height, width]
    max_indices = tf.argmax(prediction, axis=1)  
    one_hot = tf.one_hot(max_indices, depth=prediction.shape[1], axis=1, dtype=tf.float32)
    
    # Map one-hot encoding to actual class values [1, 2, 3].
    values = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    values = tf.reshape(values, (1, prediction.shape[1], 1, 1, 1))
    
    # Compute one-channel prediction (sr_seg) and target.
    one_channel_output = tf.reduce_sum(one_hot * values, axis=1)    # shape: [batch, depth, height, width]
    one_channel_target = tf.reduce_sum(tar * values, axis=1)        # shape: [batch, depth, height, width]
    
    # Convert the first sample in the batch to NumPy.
    low_res_vol = test_input[0].numpy()            # shape: [INPUT_CHANNELS, depth, height, width]
    target_vol  = one_channel_target[0].numpy()    # shape: [depth, height, width]
    output_vol  = one_channel_output[0].numpy()    # shape: [depth, height, width]
    

    tps, tns, fps, fns = [], [], [], []
    for class_val in [1.0, 2.0, 3.0]:
        pred_bin = (output_vol == class_val)
        true_bin = (target_vol  == class_val)
        TP = np.sum( pred_bin &  true_bin)
        TN = np.sum(~pred_bin & ~true_bin)
        FP = np.sum( pred_bin & ~true_bin)
        FN = np.sum(~pred_bin &  true_bin)
        tps.append(int(TP));   tns.append(int(TN))
        fps.append(int(FP));   fns.append(int(FN))

    # Build output file paths.
    if not val:
        base_out = os.path.join(OUTPUTPATH_tif)
        base_png = os.path.join(OUTPUTPATH_png)
    else:
        base_out = os.path.join(OUTPUTPATH_tif + '_val')
        base_png = os.path.join(OUTPUTPATH_png + '_val')
        
    # Define specific subfolders.
    low_res_path = os.path.join(base_out, 'low_res', f'low_res_{id}.tif')
    target_path  = os.path.join(base_out, 'target', f'target_{id}.tif')
    output_path  = os.path.join(base_out, 'output', f'output_{id}.tif')
    low_res_png_path = os.path.join(base_png, 'low_res', f'low_res_{id}.png')
    target_png_path = os.path.join(base_png, 'target', f'target{id}.png')
    output_png_path = os.path.join(base_png, 'output', f'output{id}.png')
    
    # Create directories if they do not exist.
    os.makedirs(os.path.dirname(low_res_path), exist_ok=True)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(low_res_png_path), exist_ok=True)
    os.makedirs(os.path.dirname(target_png_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    
    # Save the full 3D volumes using tifffile.
    tifffile.imwrite(low_res_path, low_res_vol)
    tifffile.imwrite(target_path, target_vol)
    tifffile.imwrite(output_path, output_vol)
    
    # Save a PNG for quick visualization: use the first slice along the depth dimension.
    low_res_first_slice = low_res_vol[0, 0, :, :]    # shape: [height, width]
    target_first_slice = target_vol[0, :, :]         # shape: [height, width]
    output_first_slice = output_vol[0, :, :]         # shape: [height, width]

    # Convert slices to uint8.
    low_res_uint8 = convert_to_uint8(low_res_first_slice)
    target_uint8  = convert_to_uint8(target_first_slice)
    output_uint8  = convert_to_uint8(output_first_slice)

    imageio.imwrite(low_res_png_path, low_res_uint8)
    imageio.imwrite(target_png_path, target_uint8)
    imageio.imwrite(output_png_path, output_uint8)

    return tps, tns, fps, fns





generator = Generator()
discriminator = Discriminator()


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)


checkpoint_dir = HOMEPATH + FOLDERPATH + '/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)





if __name__ == "__main__":


    log_dir = HOMEPATH + FOLDERPATH + '/logs'
    summary_writer = tf.summary.create_file_writer(log_dir + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


    @tf.function
    def train_step(input_image, target, step):
        """
        Expects input_image and target in [batch, channels, depth, height, width]
        (i.e. channels-first) format as required by the 3D convolution models.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            # Compute generator and discriminator losses.
            gen_total_loss, gen_gan_loss, gen_mse_loss, gen_bce_loss, dice_per_class = \
                generator_loss(disc_generated_output, gen_output, target, input_image)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # Compute gradients and apply to optimizers.
        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        # Compute metrics for Pores
        TP, TN, FP, FN = get_class2_metrics(target, gen_output)

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // train_num)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // train_num)
            tf.summary.scalar('gen_mse_loss', gen_mse_loss, step=step // train_num)
            tf.summary.scalar('gen_bce_loss', gen_bce_loss, step=step // train_num)
            tf.summary.scalar('disc_loss', disc_loss, step=step // train_num)
            for i in range(OUTPUT_CHANNELS):  
                tf.summary.scalar(f'dice_class_{i}', dice_per_class[i], step=step // train_num)

            tf.summary.scalar('TP_class_pore', TP, step=step // train_num)
            tf.summary.scalar('TN_class_pore', TN, step=step // train_num)
            tf.summary.scalar('FP_class_pore', FP, step=step // train_num)
            tf.summary.scalar('FN_class_pore', FN, step=step // train_num)

        return gen_total_loss, gen_gan_loss, gen_mse_loss, gen_bce_loss, disc_loss, dice_per_class, TP, TN, FP, FN




    def fit(train_ds, test_ds, steps):

        # For 3D volumes, we assume the raw dataset is in NDHWC format, so we need to transpose to NCDHW (channels-first) for the model.
        example_input, example_target = next(iter(test_ds.take(1)))
        example_input = tf.transpose(example_input, perm=[0, 4, 1, 2, 3])   # NDHWC -> NCDHW
        example_target = tf.transpose(example_target, perm=[0, 4, 1, 2, 3]) # NDHWC -> NCDHW

        start = time.time()

        # Create a log file
        loss_log_path = HOMEPATH + FOLDERPATH + "/epoch_loss_log.txt"
        with open(loss_log_path, "w") as f:
            f.write("Epoch,gen_total_loss,gen_gan_loss,gen_mse_loss,gen_bce_loss,disc_loss,"
                    "dice_class0,dice_class1,dice_class2,TP_class2,TN_class2,FP_class2,FN_class2,"
                    "precision_class2,recall_class2,specificity_class2\n")
        # Initialize accumulators
        gen_total_acc = gen_gan_acc = gen_mse_acc = gen_bce_acc = disc_acc = batch_count = 0
        dice_acc = np.zeros(OUTPUT_CHANNELS)  
        TP_acc = TN_acc = FP_acc = FN_acc = 0

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():

            # Transpose the input tensors from NDHWC (standard TF image format) to NCDHW for 3D
            input_image = tf.transpose(input_image, perm=[0, 4, 1, 2, 3])
            target = tf.transpose(target, perm=[0, 4, 1, 2, 3])
            
            # Call train_step and unpack losses
            gen_total_loss, gen_gan_loss, gen_mse_loss, gen_bce_loss, disc_loss, \
            dice_per_class, TP, TN, FP, FN = train_step(input_image, target, step)

            # Accumulate the losses and metrics.
            gen_total_acc += gen_total_loss.numpy()
            gen_gan_acc += gen_gan_loss.numpy()
            gen_mse_acc += gen_mse_loss.numpy()
            gen_bce_acc += gen_bce_loss.numpy()
            disc_acc += disc_loss.numpy()
            dice_acc += dice_per_class.numpy() 
            TP_acc += TP.numpy()
            TN_acc += TN.numpy()
            FP_acc += FP.numpy()
            FN_acc += FN.numpy()
            epsilon = 1e-7
            precision = TP_acc / (TP_acc + FP_acc + epsilon)
            recall    = TP_acc / (TP_acc + FN_acc + epsilon)
            specificity = TN_acc / (TN_acc + FP_acc + epsilon)

            batch_count += 1

            if (step + 1) % train_num == 0:
                epoch = (step + 1) // train_num

                avg_gen_total = gen_total_acc / batch_count
                avg_gen_gan = gen_gan_acc / batch_count
                avg_gen_mse = gen_mse_acc / batch_count
                avg_gen_bce = gen_bce_acc / batch_count
                avg_disc = disc_acc / batch_count
                avg_dice_per_class = dice_acc / batch_count

                print("\n==============================================")
                print(f"\nEpoch {epoch} Summary: \n")
                print(f"Avg gen_total_loss: {avg_gen_total:.4f}")
                print(f"Avg gen_gan_loss:   {avg_gen_gan:.4f}")
                print(f"Avg gen_mse_loss:   {avg_gen_mse:.4f}")
                print(f"Avg gen_bce_loss:   {avg_gen_bce:.4f}")
                print(f"Avg disc_loss:      {avg_disc:.4f}")
                print(f"Avg DICE Background (class 0):  {avg_dice_per_class[0]:.4f}")
                print(f"Avg DICE Eggshell (class 1):    {avg_dice_per_class[1]:.4f}")
                print(f"Avg DICE Pores (class 2):       {avg_dice_per_class[2]:.4f}\n")
                print(f"Pores performance: \nTP: {TP_acc}, TN: {TN_acc}, FP: {FP_acc}, FN: {FN_acc}")
                print(f"Precision:   {precision:.4f}")
                print(f"Recall:      {recall:.4f}")
                print(f"Specificity: {specificity:.4f}\n")

                with open(loss_log_path, "a") as f:
                    f.write(f"{epoch},{avg_gen_total:.6f},{avg_gen_gan:.6f},{avg_gen_mse:.6f},{avg_gen_bce:.6f},{avg_disc:.6f},"
                            f"{avg_dice_per_class[0]:.6f},{avg_dice_per_class[1]:.6f},{avg_dice_per_class[2]:.6f},"
                            f"{TP_acc},{TN_acc},{FP_acc},{FN_acc},"
                            f"{precision:.6f},{recall:.6f},{specificity:.6f}\n")
                
                # Reset accumulators for the next epoch
                gen_total_acc = gen_gan_acc = gen_mse_acc = gen_bce_acc = disc_acc = batch_count = 0
                dice_acc = np.zeros(OUTPUT_CHANNELS) 

                display.clear_output(wait=True)
                print(f'Time taken for epoch {epoch}: {time.time() - start:.2f} sec\n')
                print("==============================================\n\n\n")
                start = time.time()

                # (Optional) Generate sample output images using the example input from the test dataset.
                # generate_images(generator, example_input, example_target, epoch, val=False)

                checkpoint.save(file_prefix=checkpoint_prefix)

    # Call train steps
    fit(train_dataset, test_dataset, steps=train_num * EPOCH) 