"""
Exercise 2.4: Channel GAN Implementation

This script simulates a Rayleigh channel using a Conditional Generative Adversarial Network (CGAN).
It attempts to learn the channel distribution without explicit channel information.

TODO:
Complete the `generate_real_samples_with_labels_Rayleigh` function. You need to:
    1. Randomly select channel coefficients from the dataset.
    2. Generate random QAM symbols.
    3. Simulate the received signal passing through the Rayleigh channel with noise.
    4. Construct the conditioning vector.
"""

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import scipy.io as sio 
tf.disable_v2_behavior()

# ────────────────────────  Configuration  ────────────────────────────────── #
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

""" This file is trying to simulate the Rayleigh channel without any channel information"""

tf.set_random_seed(100)
np.random.seed(100)


# ────────────────────────  Model Definition  ──────────────────────────────── #
def generator_conditional(z, conditioning):  # need to change the structure
    """
    Build the Generator network for CGAN.
    
    Args:
        z: Noise vector.
        conditioning: Conditioning vector (e.g., transmitted symbol, pilot).
        
    Returns:
        G_logit: Output of the generator (simulated received signal).
    """
    z_combine = tf.concat([z, conditioning], 1)
    G_h1 = tf.nn.relu(tf.matmul(z_combine, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_logit = tf.matmul(G_h3, G_W4) + G_b4
    return G_logit


def discriminator_conditional(X, conditioning):  # need to change the structure
    """
    Build the Discriminator network for CGAN.
    
    Args:
        X: Input sample (real or generated).
        conditioning: Conditioning vector.
        
    Returns:
        D_prob: Probability that the input is real.
        D_logit: Raw output logits of the discriminator.
    """
    z_combine = tf.concat([X, conditioning], 1)
    D_h1_real = tf.nn.relu(tf.matmul(z_combine / 4, D_W1) + D_b1)
    D_h2_real = tf.nn.relu(tf.matmul(D_h1_real, D_W2) + D_b2)
    D_h3_real = tf.nn.relu(tf.matmul(D_h2_real, D_W3) + D_b3)
    D_logit = tf.matmul(D_h3_real, D_W4) + D_b4
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


# ────────────────────────  Data Preparation (TODO) ──────────────────────────────── #
def sample_Z(sample_size):
    ''' Sampling the generation noise Z from normal distribution '''
    return np.random.normal(size=sample_size)


def xavier_init(size):
    """
    Xavier initialization for network weights.

    Args:
        size: Shape of the weight tensor.

    Returns:
        Tensor initialized with Xavier distribution.
    """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def generate_real_samples_with_labels_Rayleigh(h_dataset, number=100):
    """
    Generate real (labeled) samples for training the CGAN.
    
    TODO:
        Perform the following steps:
        1. Select random channel coefficients from `h_dataset`.
        2. Generate random QAM symbols (QPSK, 16QAM, etc.).
        3. Create the received signal (y = hx + n).
        4. Construct the conditioning vector.

    Args:
        h_dataset: Dataset of channel coefficients.
        number: Number of samples to generate.
        
    Returns:
        received_data: Simulated received signal (real + imag part).
        conditioning: Vector containing labels and channel information.
    """
    # np.random.choice select 'number' samples from h_dataset
    # TO DO: realize the function
    return received_data, conditioning


# ────────────────────────  Main Execution  ────────────────────────────────── #
""" ==== Here is the main function ==== """
mean_set_QAM = np.asarray([-3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j, -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
                           1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j, 3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j
                           ], dtype=np.complex64)

# load .mat dataset
mat_file_path = 'rayleigh_channel_dataset.mat'  # file name
mat_data = sio.loadmat(mat_file_path)
h_dataset = mat_data['h_siso'].flatten()

batch_size = 512
condition_depth = 2
condition_dim = 4
Z_dim = 16
model = 'ChannelGAN_Rayleigh_'
data_size = 10000

# train by the dataset loaded
data, one_hot_labels = generate_real_samples_with_labels_Rayleigh(h_dataset, data_size)

# ────────────────────────  Graph Construction  ───────────────────────────── #
D_W1 = tf.Variable(xavier_init([2 + condition_dim, 32]))
D_b1 = tf.Variable(tf.zeros(shape=[32]))
D_W2 = tf.Variable(xavier_init([32, 32]))
D_b2 = tf.Variable(tf.zeros(shape=[32]))
D_W3 = tf.Variable(xavier_init([32, 32]))
D_b3 = tf.Variable(tf.zeros(shape=[32]))
D_W4 = tf.Variable(xavier_init([32, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3, D_W4, D_b4]
G_W1 = tf.Variable(xavier_init([Z_dim + condition_dim, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))
G_W2 = tf.Variable(xavier_init([128, 128]))
G_b2 = tf.Variable(tf.zeros(shape=[128]))
G_W3 = tf.Variable(xavier_init([128, 128]))
G_b3 = tf.Variable(tf.zeros(shape=[128]))
G_W4 = tf.Variable(xavier_init([128, 2]))
G_b4 = tf.Variable(tf.zeros(shape=[2]))
theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3, G_W4, G_b4]
R_sample = tf.placeholder(tf.float32, shape=[None, 2])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
Condition = tf.placeholder(tf.float32, shape=[None, condition_dim])
G_sample = generator_conditional(Z, Condition)
D_prob_real, D_logit_real = discriminator_conditional(R_sample, Condition)
D_prob_fake, D_logit_fake = discriminator_conditional(G_sample, Condition)

# ────────────────────────  Loss & Optimization  ────────────────────────── #
D_loss = tf.reduce_mean(D_logit_fake) - tf.reduce_mean(D_logit_real)
G_loss = -1 * tf.reduce_mean(D_logit_fake)
lambdda = 5
alpha = tf.random_uniform(shape=tf.shape(R_sample), minval=0., maxval=1.)
differences = G_sample - R_sample
interpolates = R_sample + (alpha * differences)
_, D_inter = discriminator_conditional(interpolates, Condition)
gradients = tf.gradients(D_inter, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
D_loss += lambdda * gradient_penalty
D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss, var_list=theta_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

save_fig_path = model + "images"
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)
i = 0
plt.figure(figsize=(5, 5))
plt.plot(data[:1000, 0], data[:1000, 1], 'b.')
# axes = plt.gca()
# axes.set_xlim([-4, 4])
# axes.set_ylim([-4, 4])
# plt.title('True data distribution')
# plt.savefig(save_fig_path + '/real.png', bbox_inches='tight')
np_samples = []
plot_every = 1000
plt.figure(figsize=(5, 5))
xmax = 4
saver = tf.train.Saver()

# ────────────────────────  Training Loop  ──────────────────────────────── #
for it in range(750000):
    start_idx = it * batch_size % data_size
    if start_idx + batch_size >= len(data):
        continue
    X_mb = data[start_idx:start_idx + batch_size, :]
    one_hot_labels_mb = one_hot_labels[start_idx:start_idx + batch_size, :]
    for d_idx in range(10):
        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={R_sample: X_mb, Z: sample_Z((batch_size, Z_dim)),
                                             Condition: one_hot_labels_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={R_sample: X_mb, Z: sample_Z((batch_size, Z_dim)),
                                         Condition: one_hot_labels_mb})

    if (it + 1) % plot_every == 0:
        save_path = saver.save(sess, './Models/ChannelGAN_model_step_' + str(it) + '.ckpt')

        print("Start Plotting")
        colors = ['b.', 'r+', 'm.', 'c.', 'k.', 'g.', 'y.', 'm.', \
                  'bo', 'ro', 'mo', 'co', 'ko', 'go', 'yo', 'bo']
        colors = ['b.', 'b+', 'bx', 'b^', 'b^', 'bx', 'b+', 'b.', \
                  'b.', 'b+', 'bx', 'b^', 'b^', 'bx', 'b+', 'b.']
        plt.clf()
        samples = np.array([])
        for channel_idx in range(10):
            plt.clf()
            number = 20

            # randomly choose a channel from dataset for plot
            h_complex_for_plot = np.random.choice(h_dataset, 1)[0]

            h_r = np.tile(np.real(h_complex_for_plot), number)
            h_i = np.tile(np.imag(h_complex_for_plot), number)

            for idx in range(len(mean_set_QAM)):
                labels_index = np.tile(idx, number)
                h_complex = h_r + 1j * h_i
                data_t = mean_set_QAM[labels_index]
                transmit_data = h_complex * data_t
                transmit_data = np.hstack((np.real(transmit_data).reshape(len(transmit_data), 1),
                                           np.imag(transmit_data).reshape(len(transmit_data), 1)))
                gaussion_random = np.random.multivariate_normal([0, 0], [[0.03, 0], [0, 0.03]], number).astype(
                    np.float32)
                received_data = transmit_data + gaussion_random
                conditioning = np.hstack(
                    (np.real(data_t).reshape(len(data_t), 1), np.imag(data_t).reshape(len(data_t), 1),
                     h_r.reshape(len(data_t), 1), h_i.reshape(len(data_t), 1))) / 3
                samples_component = sess.run(G_sample,
                                             feed_dict={Z: sample_Z((number, Z_dim)), Condition: conditioning})
                plt.plot(samples_component[:, 0], samples_component[:, 1], colors[idx])
                plt.plot(transmit_data[:, 0], transmit_data[:, 1], colors[idx])
            axes = plt.gca()
            axes.set_xlim([-4, 4])
            axes.set_ylim([-4, 4])
            xlabel = r'$Re\{y_n\}$'
            ylabel = r'$Imag\{y_n\}$'
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
            plt.savefig(save_fig_path + '/' + str(channel_idx) + '_{}_noise_1.eps'.format(str(i).zfill(3)),
                        bbox_inches='tight')
            plt.savefig(save_fig_path + '/' + str(channel_idx) + '_{}_noise_1.png'.format(str(i).zfill(3)),
                        bbox_inches='tight')

        axes.set_xlim([-4, 4])
        axes.set_ylim([-4, 4])
        plt.title('Iter: {}, loss(D): {:2.2f}, loss(G):{:2.2f}'.format(it + 1, D_loss_curr, G_loss_curr))
        plt.savefig(save_fig_path + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')

        i += 1
