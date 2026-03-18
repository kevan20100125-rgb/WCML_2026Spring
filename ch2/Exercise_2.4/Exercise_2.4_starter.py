"""
Exercise 2.4: Channel GAN Implementation

This script trains a CGAN to model the received-signal distribution of a
Rayleigh-like fading channel generated from a QuaDRiGa dataset.

What I want this script to do:
- load channel snapshots from rayleigh_channel_dataset.mat
- generate training pairs using y = h*x + n
- use [Re(x), Im(x), Re(h), Im(h)] as the conditioning vector
- normalize both y and conditioning before training
- try GPU first, and fall back to CPU if no usable GPU is available
- evaluate every checkpoint on a fixed evaluation set
- keep track of the best checkpoint automatically
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# -----------------------------------------------------------------------------
# basic setup
# -----------------------------------------------------------------------------
tf.set_random_seed(100)
np.random.seed(100)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -----------------------------------------------------------------------------
# device setup: try GPU first, otherwise just use CPU
# -----------------------------------------------------------------------------
physical_gpus = tf.config.list_physical_devices('GPU')

if physical_gpus:
    try:
        tf.config.set_visible_devices(physical_gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_gpus[0], True)
        print("Using GPU:", physical_gpus[0])
    except Exception as e:
        print("GPU setup failed. Falling back to CPU. Reason:", e)
else:
    print("No GPU detected. Running on CPU.")

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def sample_Z(sample_size):
    """sample latent noise from a standard normal"""
    return np.random.normal(size=sample_size).astype(np.float32)


def xavier_init(size):
    """simple Xavier init for dense layers"""
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def lrelu(x, alpha=0.2):
    """leaky ReLU tends to be a bit safer than plain ReLU here"""
    return tf.maximum(alpha * x, x)


def generate_real_samples_with_labels_Rayleigh(h_dataset, number=100):
    """
    build real training samples for the CGAN

    what I want here:
    - randomly pick a channel h from the dataset
    - randomly pick a 16-QAM symbol x
    - form y = h*x + n
    - build conditioning as [Re(x), Im(x), Re(h), Im(h)]
    """

    # randomly pick channels from the saved QuaDRiGa dataset
    h_complex = np.random.choice(h_dataset, size=number)

    # randomly pick transmitted 16-QAM symbols
    symbol_indices = np.random.randint(0, len(mean_set_QAM), size=number)
    data_t = mean_set_QAM[symbol_indices]

    # noiseless received signal
    noiseless_rx = h_complex * data_t

    # keep noise tiny because the channel magnitude is around 1e-5
    gaussian_random = np.random.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[1e-12, 0.0], [0.0, 1e-12]],
        size=number
    ).astype(np.float32)

    # split complex y into [Re(y), Im(y)]
    received_data = np.hstack((
        np.real(noiseless_rx).reshape(number, 1),
        np.imag(noiseless_rx).reshape(number, 1)
    )).astype(np.float32)

    received_data = received_data + gaussian_random

    # conditioning = transmitted symbol + channel coefficient
    conditioning = np.hstack((
        np.real(data_t).reshape(number, 1),
        np.imag(data_t).reshape(number, 1),
        np.real(h_complex).reshape(number, 1),
        np.imag(h_complex).reshape(number, 1)
    )).astype(np.float32)

    return received_data, conditioning


def normalize_array(x, mean, std):
    return (x - mean) / std


def denormalize_array(x, mean, std):
    return x * std + mean


def build_training_pool(h_dataset, number):
    """
    generate one fixed training pool first
    then compute normalization stats from this pool
    """
    raw_y, raw_c = generate_real_samples_with_labels_Rayleigh(h_dataset, number)

    y_mean = raw_y.mean(axis=0, keepdims=True)
    y_std = raw_y.std(axis=0, keepdims=True) + 1e-8

    c_mean = raw_c.mean(axis=0, keepdims=True)
    c_std = raw_c.std(axis=0, keepdims=True) + 1e-8

    norm_y = normalize_array(raw_y, y_mean, y_std)
    norm_c = normalize_array(raw_c, c_mean, c_std)

    return raw_y, raw_c, norm_y, norm_c, y_mean, y_std, c_mean, c_std


def evaluate_checkpoint(
    sess,
    checkpoint_path,
    saver,
    eval_channels,
    y_mean,
    y_std,
    c_mean,
    c_std,
    number_per_symbol=64
):
    """
    evaluate one checkpoint using fixed channels

    lower score is better
    """
    saver.restore(sess, checkpoint_path)

    center_errors = []
    spread_errors = []

    for h_complex_for_eval in eval_channels:
        h_r = np.tile(np.real(h_complex_for_eval), number_per_symbol)
        h_i = np.tile(np.imag(h_complex_for_eval), number_per_symbol)

        for idx in range(len(mean_set_QAM)):
            labels_index = np.tile(idx, number_per_symbol)
            h_complex = h_r + 1j * h_i
            data_t = mean_set_QAM[labels_index]

            transmit_data_complex = h_complex * data_t
            transmit_data = np.hstack((
                np.real(transmit_data_complex).reshape(len(transmit_data_complex), 1),
                np.imag(transmit_data_complex).reshape(len(transmit_data_complex), 1)
            )).astype(np.float32)

            gaussian_random = np.random.multivariate_normal(
                [0.0, 0.0],
                [[1e-12, 0.0], [0.0, 1e-12]],
                number_per_symbol
            ).astype(np.float32)

            received_data = transmit_data + gaussian_random

            conditioning_raw = np.hstack((
                np.real(data_t).reshape(len(data_t), 1),
                np.imag(data_t).reshape(len(data_t), 1),
                h_r.reshape(len(data_t), 1),
                h_i.reshape(len(data_t), 1)
            )).astype(np.float32)

            conditioning_norm = normalize_array(conditioning_raw, c_mean, c_std)

            fake_norm = sess.run(
                G_sample,
                feed_dict={
                    Z: sample_Z((number_per_symbol, Z_dim)),
                    Condition: conditioning_norm
                }
            )

            fake = denormalize_array(fake_norm, y_mean, y_std)

            # cluster center error
            real_center = received_data.mean(axis=0)
            fake_center = fake.mean(axis=0)
            center_err = np.linalg.norm(real_center - fake_center)

            # cluster spread error
            real_std = received_data.std(axis=0)
            fake_std = fake.std(axis=0)
            spread_err = np.linalg.norm(real_std - fake_std)

            center_errors.append(center_err)
            spread_errors.append(spread_err)

    score = np.mean(center_errors) + 0.5 * np.mean(spread_errors)
    return score, np.mean(center_errors), np.mean(spread_errors)



# -----------------------------------------------------------------------------
# main setup
# -----------------------------------------------------------------------------
mean_set_QAM = np.asarray([
    -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
    -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
     1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,
     3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j
], dtype=np.complex64)

mat_file_path = 'rayleigh_channel_dataset.mat'
mat_data = sio.loadmat(mat_file_path)
h_dataset = mat_data['h_siso'].flatten()

print("Loaded channel dataset:", h_dataset.shape, h_dataset.dtype)

batch_size = 512
condition_dim = 4
Z_dim = 16
model = 'ChannelGAN_Rayleigh_'
data_size = 20000

# build one fixed training pool first
raw_data, raw_condition, data, cond_data, y_mean, y_std, c_mean, c_std = build_training_pool(
    h_dataset, data_size
)

print("Normalized training pool built.")
print("y_mean:", y_mean)
print("y_std :", y_std)
print("c_mean:", c_mean)
print("c_std :", c_std)

# fix evaluation channels for fair checkpoint comparison
eval_seed = 2026
eval_rng = np.random.RandomState(eval_seed)
num_eval_channels = 5
eval_channels = eval_rng.choice(h_dataset, size=num_eval_channels, replace=False)

print("Fixed evaluation channels prepared:", num_eval_channels)

# -----------------------------------------------------------------------------
# model definition
# -----------------------------------------------------------------------------
def generator_conditional(z, conditioning):
    """
    generator:
    take noise z + conditioning and generate a normalized 2D received sample
    """
    z_combine = tf.concat([z, conditioning], axis=1)
    G_h1 = lrelu(tf.matmul(z_combine, G_W1) + G_b1)
    G_h2 = lrelu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = lrelu(tf.matmul(G_h2, G_W3) + G_b3)
    G_logit = tf.matmul(G_h3, G_W4) + G_b4
    return G_logit


def discriminator_conditional(X, conditioning):
    """
    discriminator:
    take a normalized sample + normalized conditioning and decide whether it looks real
    """
    z_combine = tf.concat([X, conditioning], axis=1)
    D_h1_real = lrelu(tf.matmul(z_combine, D_W1) + D_b1)
    D_h2_real = lrelu(tf.matmul(D_h1_real, D_W2) + D_b2)
    D_h3_real = lrelu(tf.matmul(D_h2_real, D_W3) + D_b3)
    D_logit = tf.matmul(D_h3_real, D_W4) + D_b4
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


# -----------------------------------------------------------------------------
# graph construction
# -----------------------------------------------------------------------------
D_W1 = tf.Variable(xavier_init([2 + condition_dim, 64]))
D_b1 = tf.Variable(tf.zeros(shape=[64]))
D_W2 = tf.Variable(xavier_init([64, 64]))
D_b2 = tf.Variable(tf.zeros(shape=[64]))
D_W3 = tf.Variable(xavier_init([64, 64]))
D_b3 = tf.Variable(tf.zeros(shape=[64]))
D_W4 = tf.Variable(xavier_init([64, 1]))
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

# -----------------------------------------------------------------------------
# loss and optimization
# -----------------------------------------------------------------------------
D_loss = tf.reduce_mean(D_logit_fake) - tf.reduce_mean(D_logit_real)
G_loss = -1.0 * tf.reduce_mean(D_logit_fake)

lambdda = 10.0
alpha = tf.random_uniform(shape=tf.shape(R_sample), minval=0.0, maxval=1.0)
differences = G_sample - R_sample
interpolates = R_sample + (alpha * differences)
_, D_inter = discriminator_conditional(interpolates, Condition)
gradients = tf.gradients(D_inter, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
D_loss += lambdda * gradient_penalty

D_solver = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.0, beta2=0.9
).minimize(D_loss, var_list=theta_D)

G_solver = tf.train.AdamOptimizer(
    learning_rate=1e-4, beta1=0.0, beta2=0.9
).minimize(G_loss, var_list=theta_G)

session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.allow_growth = True
sess = tf.Session(config=session_config)
sess.run(tf.global_variables_initializer())

# -----------------------------------------------------------------------------
# output folders
# -----------------------------------------------------------------------------
save_fig_path = model + "images"
save_model_path = "Models"

if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)


print("Figure output directory:", os.path.abspath(save_fig_path))
print("Model output directory:", os.path.abspath(save_model_path))

# quick look at the real data cloud
plt.figure(figsize=(5, 5))
plt.plot(raw_data[:1000, 0], raw_data[:1000, 1], 'b.', markersize=4)
plt.xlabel(r'$Re\{y_n\}$')
plt.ylabel(r'$Imag\{y_n\}$')
plt.title('Real data preview')
plt.savefig(os.path.join(save_fig_path, 'real_preview.png'), dpi=200, bbox_inches='tight')
plt.close()

plot_every = 2000
i = 0
saver = tf.train.Saver()

best_score = np.inf
best_checkpoint = None
best_iter = None

# -----------------------------------------------------------------------------
# minibatch index helper
# -----------------------------------------------------------------------------
perm = np.random.permutation(data_size)
cursor = 0

def next_batch(batch_size):
    global perm, cursor
    if cursor + batch_size > data_size:
        perm = np.random.permutation(data_size)
        cursor = 0
    idx = perm[cursor:cursor + batch_size]
    cursor += batch_size
    return data[idx], cond_data[idx]

# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------
num_iterations = 10000

for it in range(num_iterations):
    X_mb, Condition_mb = next_batch(batch_size)

    # this is enough for this toy problem; 10 critic steps is overkill here
    for _ in range(5):
        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={
                R_sample: X_mb,
                Z: sample_Z((batch_size, Z_dim)),
                Condition: Condition_mb
            }
        )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={
            R_sample: X_mb,
            Z: sample_Z((batch_size, Z_dim)),
            Condition: Condition_mb
        }
    )

    if (it + 1) % 500 == 0:
        print("Iter: {}, loss(D): {:.6f}, loss(G): {:.6f}".format(
            it + 1, D_loss_curr, G_loss_curr
        ))

    if (it + 1) % plot_every == 0:
        save_path = saver.save(
            sess,
            os.path.join(save_model_path, 'ChannelGAN_model_step_{}.ckpt'.format(it + 1))
        )
        print("Checkpoint saved to:", save_path)
        print("Start Plotting")

        # keep the original plotting for qualitative inspection
        for channel_idx in range(10):
            plt.figure(figsize=(5, 5))
            number = 40

            # randomly choose one channel for this plot
            h_complex_for_plot = np.random.choice(h_dataset, 1)[0]
            h_r = np.tile(np.real(h_complex_for_plot), number)
            h_i = np.tile(np.imag(h_complex_for_plot), number)

            all_real_x = []
            all_real_y = []
            all_fake_x = []
            all_fake_y = []

            for idx in range(len(mean_set_QAM)):
                labels_index = np.tile(idx, number)
                h_complex = h_r + 1j * h_i
                data_t = mean_set_QAM[labels_index]

                transmit_data_complex = h_complex * data_t
                transmit_data = np.hstack((
                    np.real(transmit_data_complex).reshape(len(transmit_data_complex), 1),
                    np.imag(transmit_data_complex).reshape(len(transmit_data_complex), 1)
                )).astype(np.float32)

                gaussian_random = np.random.multivariate_normal(
                    [0.0, 0.0],
                    [[1e-12, 0.0], [0.0, 1e-12]],
                    number
                ).astype(np.float32)

                received_data = transmit_data + gaussian_random

                conditioning_raw = np.hstack((
                    np.real(data_t).reshape(len(data_t), 1),
                    np.imag(data_t).reshape(len(data_t), 1),
                    h_r.reshape(len(data_t), 1),
                    h_i.reshape(len(data_t), 1)
                )).astype(np.float32)

                conditioning_norm = normalize_array(conditioning_raw, c_mean, c_std)

                samples_component_norm = sess.run(
                    G_sample,
                    feed_dict={
                        Z: sample_Z((number, Z_dim)),
                        Condition: conditioning_norm
                    }
                )

                samples_component = denormalize_array(samples_component_norm, y_mean, y_std)

                all_real_x.append(received_data[:, 0])
                all_real_y.append(received_data[:, 1])
                all_fake_x.append(samples_component[:, 0])
                all_fake_y.append(samples_component[:, 1])

                # real points: blue circles
                plt.plot(
                    received_data[:, 0], received_data[:, 1],
                    'bo', markersize=3, alpha=0.55
                )

                # generated points: red x
                plt.plot(
                    samples_component[:, 0], samples_component[:, 1],
                    'rx', markersize=4, alpha=0.75
                )

            all_real_x = np.concatenate(all_real_x)
            all_real_y = np.concatenate(all_real_y)
            all_fake_x = np.concatenate(all_fake_x)
            all_fake_y = np.concatenate(all_fake_y)

            x_min = min(all_real_x.min(), all_fake_x.min())
            x_max = max(all_real_x.max(), all_fake_x.max())
            y_min = min(all_real_y.min(), all_fake_y.min())
            y_max = max(all_real_y.max(), all_fake_y.max())

            x_margin = max((x_max - x_min) * 0.25, 1e-6)
            y_margin = max((y_max - y_min) * 0.25, 1e-6)

            axes = plt.gca()
            axes.set_xlim([x_min - x_margin, x_max + x_margin])
            axes.set_ylim([y_min - y_margin, y_max + y_margin])

            plt.xlabel(r'$Re\{y_n\}$')
            plt.ylabel(r'$Imag\{y_n\}$')
            plt.title('Iter: {}, ChIdx: {}, D: {:.4f}, G: {:.4f}'.format(
                it + 1, channel_idx, D_loss_curr, G_loss_curr
            ))

            png_path = os.path.join(
                save_fig_path,
                '{}_{}_noise_1.png'.format(channel_idx, str(i).zfill(3))
            )
            eps_path = os.path.join(
                save_fig_path,
                '{}_{}_noise_1.eps'.format(channel_idx, str(i).zfill(3))
            )

            plt.savefig(png_path, dpi=200, bbox_inches='tight')
            plt.savefig(eps_path, bbox_inches='tight')
            plt.close()

        # evaluate this checkpoint on fixed channels
        score, center_score, spread_score = evaluate_checkpoint(
            sess,
            save_path,
            saver,
            eval_channels,
            y_mean,
            y_std,
            c_mean,
            c_std,
            number_per_symbol=64
        )

        print(
            "Eval score -> total: {:.8e}, center: {:.8e}, spread: {:.8e}".format(
                score, center_score, spread_score
            )
        )

        if score < best_score:
            best_score = score
            best_checkpoint = save_path
            best_iter = it + 1
            print("New best checkpoint at iter {} with score {:.8e}".format(
                best_iter, best_score
            ))

        i += 1

print("Training finished.")
print("Best checkpoint:", best_checkpoint)
print("Best iteration :", best_iter)
print("Best score     :", best_score)