import tensorflow as tf
import numpy as np
# Disable eager execution (for compatibility with older TF APIs)
tf.compat.v1.disable_eager_execution()

def block_error_ratio_vae_awgn(snrs_db, block_size, channel_use, batch_size, nrof_steps):
    
    print('block_size %d'%(block_size))
    print('channel_use %d'%(channel_use))
    
    # Amount of info transmitted per channel use
    rate = float(block_size)/float(channel_use)
    print('rate %0.2f'%(rate))
    
    # The input is one-hot encoded vector for each codeword
    # Total number of codewords = pow(2,block_size)
    alphabet_size = pow(2, block_size)
    alphabet = np.eye(alphabet_size, dtype='float32')  # One-hot encoded values
    
    # Repeat the alphabet to create training and test datasets
    train_dataset = np.transpose(np.tile(alphabet, int(batch_size)))
    test_dataset = np.transpose(np.tile(alphabet, int(batch_size * 1000)))
    
    print('--Setting up VAE graph--')
    input, output, noise_std_dev, h_norm, kl_loss = _implement_variational_autoencoder(alphabet_size, channel_use)
    
    print('--Setting up training scheme--')
    train_step = _implement_vae_training(output, input, kl_loss)
    
    print('--Setting up accuracy--')
    accuracy = _implement_accuracy(output, input)

    print('--Starting the tensorflow session--')
    sess = _setup_interactive_tf_session()
    _init_and_start_tf_session(sess)
    
    print('--Training the VAE over AWGN channel--')
    _train_vae(train_step, input, noise_std_dev, nrof_steps, train_dataset, snrs_db, rate, accuracy)
    
    print('--Evaluating VAE performance--')
    bler = _evaluate(input, noise_std_dev, test_dataset, snrs_db, rate, accuracy)
    
    print('--Closing the session--')
    _close_tf_session(sess)
    
    return bler
    
def _setup_tf_session():
    return tf.Session()

def _setup_interactive_tf_session():
    return tf.compat.v1.InteractiveSession()

def _init_and_start_tf_session():
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess

def _init_and_start_tf_session(sess):
    sess.run(tf.compat.v1.global_variables_initializer())
    
def _close_tf_session(sess):
    sess.close
    
def _weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def _implement_variational_autoencoder(input_dimension, encoder_dimension):
    input = tf.compat.v1.placeholder(tf.float32, [None, input_dimension])
    
    # Densely connected encoder layers
    W_enc1 = _weight_variable([input_dimension, input_dimension])
    b_enc1 = _bias_variable([input_dimension])
    
    h_enc1 = tf.nn.relu(tf.matmul(input, W_enc1) + b_enc1)
    
    # VAE encoder produces mean and log variance
    W_enc_mean = _weight_variable([input_dimension, encoder_dimension])
    b_enc_mean = _bias_variable([encoder_dimension])
    
    W_enc_logvar = _weight_variable([input_dimension, encoder_dimension])
    b_enc_logvar = _bias_variable([encoder_dimension])
    
    # Mean and log variance of latent distribution
    z_mean = tf.matmul(h_enc1, W_enc_mean) + b_enc_mean
    z_logvar = tf.matmul(h_enc1, W_enc_logvar) + b_enc_logvar
    
    # Reparameterization trick
    eps = tf.random.normal(tf.shape(z_mean), mean=0.0, stddev=1.0)
    z = z_mean + tf.multiply(tf.exp(0.5 * z_logvar), eps)
    
    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
    
    # Normalization layer (to enforce power constraint)
    normalization_factor = tf.math.reciprocal(tf.sqrt(tf.reduce_sum(tf.square(z), 1))) * np.sqrt(encoder_dimension)
    h_norm = tf.multiply(tf.tile(tf.expand_dims(normalization_factor, 1), [1, encoder_dimension]), z)

    # AWGN noise layer
    noise_std_dev = tf.compat.v1.placeholder(tf.float32)
    channel = tf.random.normal(tf.shape(h_norm), stddev=noise_std_dev)
    h_noisy = tf.add(h_norm, channel)
    
    # Densely connected decoder layer
    W_dec1 = _weight_variable([encoder_dimension, input_dimension])
    b_dec1 = _bias_variable([input_dimension])
    
    h_dec1 = tf.nn.relu(tf.matmul(h_noisy, W_dec1) + b_dec1)
        
    # Output layer
    W_out = _weight_variable([input_dimension, input_dimension])
    b_out = _bias_variable([input_dimension])
     
    output = tf.nn.softmax(tf.matmul(h_dec1, W_out) + b_out)
    
    return (input, output, noise_std_dev, h_norm, kl_loss)
    
def _implement_vae_training(output, input, kl_loss):
    # Reconstruction loss (cross entropy)
    recon_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=input))
    
    # Total loss is a weighted sum of reconstruction loss and KL divergence
    # Beta is a hyperparameter that controls the importance of the KL term
    beta = 0.001  # This might need tuning depending on your specific application
    total_loss = recon_loss + beta * kl_loss
    
    # Optimizer
    train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(total_loss)
    
    return train_step

def _implement_accuracy(output, input):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def _train_vae(train_step, input, noise_std_dev, nrof_steps, training_dataset, snrs_db, rate, accuracy):
    print('--Training--')
    print('number of steps %d'%(nrof_steps))
    snrs_rev = snrs_db[::-1]
    for snr in snrs_rev[0:1]:  # Train with higher SNRs first
        print('training snr %0.2f db'%(snr))
        noise = np.sqrt(1.0 / (2 * rate * pow(10, 0.1 * snr)))
        for i in range(int(nrof_steps)):
            batch = training_dataset
            np.random.shuffle(batch)
            if (i + 1) % (nrof_steps/10) == 0:  # i = 0 is the first step
                print('training step %d'%(i + 1))
            train_step.run(feed_dict={input: batch, noise_std_dev: noise})
        print('training accuracy %0.4f'%(accuracy.eval(feed_dict={input: batch, noise_std_dev: noise})))

def _evaluate(input, noise_std_dev, test_dataset, snrs_db, rate, accuracy):
    print('--Evaluating VAE performance on test dataset--')
    bler = []
    for snr in snrs_db:
        noise = np.sqrt(1.0 / (2 * rate * pow(10, 0.1 * snr)))
        acc = accuracy.eval(feed_dict={input: test_dataset, noise_std_dev: noise})
        bler.append(1.0 - acc)
    return bler

# Example usage (similar to your original code):
# snrs_db = np.arange(-4, 8.5, 0.5)
# block_size = 4  # 4 information bits
# channel_use = 7  # 7 channel uses (to match Hamming (7,4))
# batch_size = 16
# nrof_steps = 20000
# bler_vae = block_error_ratio_vae_awgn(snrs_db, block_size, channel_use, batch_size, nrof_steps)