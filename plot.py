import tensorflow as tf
from model_gan import Generator, Discriminator
from global_path import HOMEPATH


generator = Generator()
discriminator = Discriminator()


tf.keras.utils.plot_model(generator, show_shapes=True, dpi=200, to_file=HOMEPATH+'/gen.png')
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=200, to_file=HOMEPATH+'/dis.png')