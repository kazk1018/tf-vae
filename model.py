import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Lambda, Input
from tensorflow.python.keras.activations import relu, softplus, sigmoid
from tensorflow.python.keras import backend as K


def compute_loss(model, x):
    y, z_mean, z_log_var = model(x)
    reconstruction_loss = tf.reduce_sum(K.binary_crossentropy(x, y), axis=-1)
    kl = tf.reduce_sum(1.0 + z_log_var - tf.exp(z_log_var) - tf.square(z_mean), axis=-1) * -0.5
    return tf.reduce_mean(reconstruction_loss + kl)


def reparameterize(args):
    z_mean, z_log_var = args
    epsilon = tf.random_normal(shape=(z_mean.shape[0], z_mean.shape[1]), mean=0.0, stddev=1.0)
    return z_mean + epsilon * tf.exp(0.5 * z_log_var)


class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__(self)
        self.latent_dim = latent_dim

        self.linear1 = Dense(1024, kernel_initializer='he_normal')
        self.linear2 = Dense(1024, kernel_initializer='he_normal')
        self.linear3 = Dense(1024, kernel_initializer='he_normal')
        self.linear4 = Dense(2 * self.latent_dim)

    def call(self, inputs):
        x = relu(self.linear1(inputs))
        x = relu(self.linear2(x))
        x = relu(self.linear3(x))
        x = self.linear4(x)
        z_mean = x[:, :self.latent_dim]
        z_log_var = softplus(x[:, self.latent_dim:])
        return z_mean, z_log_var


class Decoder(Model):
    def __init__(self, output_size):
        super(Decoder, self).__init__(self)

        self.linear1 = Dense(1024, kernel_initializer='he_normal')
        self.linear2 = Dense(1024, kernel_initializer='he_normal')
        self.linear3 = Dense(1024, kernel_initializer='he_normal')
        self.linear4 = Dense(output_size)
        
    def call(self, inputs):
        x = relu(self.linear1(inputs))
        x = relu(self.linear2(x))
        x = relu(self.linear3(x))
        x = sigmoid(self.linear4(x))
        return x


class VAE(Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__(self)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Lambda(reparameterize)((z_mean, z_log_var))
        y = self.decoder(z)
        return y, z_mean, z_log_var
