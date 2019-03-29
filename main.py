import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from model import Encoder, Decoder, VAE, compute_loss
from tensorflow.python.keras.datasets import mnist, fashion_mnist


# from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
def plot_results(model, data, batch_size=128, latent_dim=2, model_name="vae_mnist"):
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _ = model.encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    if latent_dim == 2:
        filename = os.path.join(model_name, "digits_over_latent.png")
        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = model.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--dim', '-d', type=int, default=2, help='size of latent space')
    args = parser.parse_args()

    (x_train, _), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32).reshape((x_train.shape[0], -1))
    x_train = x_train / 255.

    x_test = x_test.astype(np.float32).reshape((x_test.shape[0], -1))
    x_test = x_test / 255.

    encoder = Encoder(latent_dim=args.dim)
    decoder = Decoder(output_size=28*28)
    vae = VAE(encoder, decoder)

    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(args.batch_size * 10).batch(args.batch_size)

    optimizer = tf.train.AdamOptimizer()

    global_step = tf.train.get_or_create_global_step()
    for epoch in range(args.epochs):
        for i, x in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss = compute_loss(vae, x)

            gradient = tape.gradient(loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(gradient, vae.trainable_variables), global_step)

            if i % 1000 == 0:
                print('Epochs: {}, Iters: {} loss: {}'.format(epoch, i, loss.numpy()))

    data = (x_test, y_test)
    plot_results(vae, data, batch_size=args.batch_size)


if __name__ == '__main__':
    tf.enable_eager_execution()

    main()
