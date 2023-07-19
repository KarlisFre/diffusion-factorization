import tensorflow as tf
import io
from data.factoring_dataset import rand_batch, extract_x

test_batch_size = 128
import matplotlib.pyplot as plt
from config import dist
import numpy as np

@tf.function
def predict(model, x):
    return model(x, training=False)

def plot_accuracy(model, length, N=100):
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    accuracy_data = []
    loss_data = []
    grad_data = []
    for t in range(N):
        noise_scale = t/N
        images, labels = rand_batch(length, False, noise_scale, test_batch_size, for_training=False)
        predictions = predict(model, images)
        test_accuracy.reset_states()
        test_accuracy(tf.argmax(labels, axis=-1), predictions)
        accuracy = test_accuracy.result()
        accuracy_data.append(accuracy)
        with tf.GradientTape() as tape:
            tape.watch(predictions)
            loss = dist.train_loss(labels, predictions, tf.constant(np.array(noise_scale), dtype = tf.float32))
            #loss = dist.train_loss_weighted(labels, predictions, tf.cast(np.array(noise_scale)[..., np.newaxis], tf.float32), weights=1.)
        grad = tape.gradient(loss, predictions)
        grad = tf.reduce_sum(grad, axis=0) # average per batch
        grad_data.append(tf.reduce_mean(tf.abs(grad)))
        loss_data.append(loss)


    figure =plt.figure()
    plt.plot(accuracy_data)
    plt.plot([0.5]*N) #baseline
    #plt.show()
    image_tf = plot_to_image(figure)
    tf.summary.image("accuracy_curve", image_tf)
    figure = plt.figure()
    plt.plot(loss_data)
    image_tf = plot_to_image(figure)
    tf.summary.image("loss_curve", image_tf)

    figure = plt.figure()
    plt.plot(grad_data)
    image_tf = plot_to_image(figure)
    tf.summary.image("grad_curve", image_tf)

    return accuracy_data


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image