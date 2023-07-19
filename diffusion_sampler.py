import tensorflow as tf
from data.factoring_dataset import rand_batch, construct_input, decode_num, to_base, extract_mulnum
from model.factoring_model import FactoringModel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from config import dist
from config import n_classes, hidden_maps

test_batch_size = 256
TEST_LENGTH = 32 # change this to try other bit-counts for integers to factor
model_dir = 'pretrained_model'

use_baseline_sampling = False
from config import in_maps
np.set_printoptions(linewidth=2000, precision=3, suppress=True)

model = FactoringModel(in_maps, hidden_maps, n_classes)

@tf.function
def predict(x):
    predictions = model(x, training=False)
    predictions = tf.nn.softmax(predictions, axis=-1)
    return predictions

# calculates if both numbers multiply to the desired result
def mul_accuracy(predictions, mulnum, verbose=True, length=TEST_LENGTH):
    predictions = tf.argmax(predictions, axis=-1)
    predictions = tf.minimum(predictions, 1)  # hack to deal with more than 2 classes
    mulnum = tf.argmax(mulnum, axis=-1).numpy()
    x1, x2 = tf.split(predictions, 2, axis=-1)
    x1 = x1.numpy()
    x2 = x2.numpy()
    correct = 0
    item_accuracy = []

    for i in range(test_batch_size):
        a_bin = x1[i]
        b_bin = x2[i]
        mul_bin = mulnum[i]
        a = decode_num(a_bin)
        b = decode_num(b_bin)
        mul = decode_num(mul_bin)
        ok_str = 'ok' if mul == a * b else 'fail'
        obtained_num = np.array(to_base(a * b, 2, length))
        if i == 0:
            if verbose: print(a, "*", b, "=", a * b, mul, ok_str)
            if verbose and mul != a * b:
                print(mul_bin)
                print(obtained_num)

        example_correct = mul == a * b
        if example_correct: correct += 1
        item_accuracy.append(1 if example_correct else 0)

    return correct / test_batch_size, item_accuracy

# the diffusion sampling process
def diffusion(N, x_initial, mul_num_hot, labels, verbose=True, prepare_image=True, length=TEST_LENGTH):
    image_data = []
    image_data1 = []
    x = x_initial
    cum_accuracy = np.zeros(test_batch_size)
    for t in range(N):
        noise_scale = 1 - t / N
        x_noisy = dist.randomized_rounding(x)
        if use_baseline_sampling: x = x_noisy
        images = construct_input(x_noisy, mul_num_hot, noise_scale, length, batch_size=test_batch_size)
        predictions = predict(images)
        accuracy, item_accuracy = mul_accuracy(predictions, mul_num_hot, verbose=verbose, length=length)
        cum_accuracy = np.maximum(cum_accuracy, item_accuracy)
        if verbose: print("accuracy:", accuracy, "cum_accuracy:", np.mean(cum_accuracy))
        if use_baseline_sampling:
            x = dist.reverse_distribution_step_thoeretic(x, predictions, noise_scale, 1 / N)
        else:
            x = dist.reverse_distribution_step(x, predictions, noise_scale, 1/N)
        if verbose: print(noise_scale)
        if verbose: print(tf.transpose(x[0]).numpy())
        if n_classes == 2:
            image_data.append(predictions[0, :, 0].numpy())
            image_data1.append(x[0, :, 0].numpy())
        else:
            image_data.append(predictions[0, :, :].numpy())
            image_data1.append(x[0, :, :].numpy())

    if prepare_image:
        fig = plt.figure(figsize=(length * 4 / 100, N * 2 / 100), dpi=100)
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, 2),  # creates 2x2 grid of axes
                         axes_pad=0,  # pad between axes in inch.
                         )
        grid[0].imshow(image_data)
        grid[1].imshow(image_data1)
        grid[0].axes.xaxis.set_visible(False)
        grid[0].axes.yaxis.set_visible(False)
        grid[1].axes.xaxis.set_visible(False)
        grid[1].axes.yaxis.set_visible(False)
        plt.show()

    return np.mean(cum_accuracy)

def sampling(diffusion_steps):
    images, labels = rand_batch(TEST_LENGTH, False, 0.0, test_batch_size, for_training=False)
    mulnum = extract_mulnum(images)

    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    initial_x = tf.ones([test_batch_size, TEST_LENGTH, n_classes]) / n_classes
    print("Sampling...")
    accuracy = diffusion(diffusion_steps, initial_x, mulnum, labels, verbose=False)
    print("Fraction of examples solved in "+str(diffusion_steps), "sampling steps:", accuracy)


sampling(256)
