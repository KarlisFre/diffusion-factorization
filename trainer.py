import datetime
import time

import numpy as np
import tensorflow as tf

from config import dist
from config import n_classes, batch_size, dropout_rate, hidden_maps
from data.factoring_dataset import create_factoring_dataset
from diffusion_utils import plot_accuracy
from model.factoring_model import FactoringModel
from model.transformer.transformer import Transformer
from optimization.AdaBelief import AdaBeliefOptimizer
from config import in_maps

EPOCHS = 500
total_saved_epochs = 20
lengths = [16, 24, 32] # the lengths to be trained on
test_length = lengths[-1]
train_iters = 100
train_dir = 'checkpoint'
load_prev = False

#distributed_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
distributed_strategy = tf.distribute.MirroredStrategy()

learning_rate = tf.constant(0.00125 * np.sqrt(96 / hidden_maps)*np.sqrt(batch_size/64), dtype=tf.float32)

train_ds_list = [create_factoring_dataset(length, variable_length=False, for_training=True, dataset_size=10000000) for length in lengths]
train_ds_list = [dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE) for dataset in train_ds_list]

test_ds = create_factoring_dataset(test_length, for_training=False, dataset_size=100000)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).take(10)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

train_ds_list = [x.with_options(options) for x in train_ds_list]
train_ds_list = [distributed_strategy.experimental_distribute_dataset(x) for x in train_ds_list]

test_ds = test_ds.with_options(options)
test_ds = distributed_strategy.experimental_distribute_dataset(test_ds)
dataset_iters = [iter(ds) for ds in train_ds_list]

# Create an instance of the model
with distributed_strategy.scope():
    model = FactoringModel(in_maps, hidden_maps, n_classes, dropout_rate)
    # model = Transformer({"encoder_only": False,
    #                      "hidden_size": hidden_maps,
    #                      "n_classes": n_classes,
    #                      "dropout_rate": dropout_rate,
    #                      "dtype": tf.float32,
    #                      "padded_decode": False,
    #                      "layer_postprocess_dropout": 0,
    #                      "num_hidden_layers": 8,
    #                      "num_heads": 8,
    #                      "attention_dropout": dropout_rate,
    #                      "filter_size": hidden_maps*4,
    #                      "relu_dropout": 0,
    #                      "extra_decode_length": 0,
    #                      "vocab_size": 2,
    #                      "beam_size": 1,
    #                      "alpha": 1})

    total_steps = EPOCHS * 100
    warmup_steps = 5000

    def has_decay(var):
        has_decay = not "residual_scale" in var.name
        return has_decay

    optimizer = AdaBeliefOptimizer(learning_rate, beta_2=0.99, epsilon=1e-7, weight_decay=0.01, has_decay_func=has_decay,
                                   clip_gradients=True, total_steps=total_steps,
                                   warmup_proportion=warmup_steps / total_steps)
    optimizer0 = optimizer
    default_optimizer_config = optimizer0.get_config()

    averager = tf.train.ExponentialMovingAverage(0.99)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    train_accuracy_adv = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_adv')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step1(data_list):
    with tf.GradientTape() as tape:
        sum_loss = 0
        n_bins = len(data_list)
        for k in range(n_bins):
            image, labels, noise_scales = data_list[k]
            predictions = model(image, training=True, log_in_tb=(k == n_bins - 1))
            loss = dist.train_loss(labels, predictions, noise_scales)
            train_accuracy(tf.argmax(labels, axis=-1), predictions)
            sum_loss += loss * np.sqrt(lengths[k]/32)
        sum_loss /= n_bins

    gradients = tape.gradient(sum_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(sum_loss)

    return sum_loss, predictions, gradients


@tf.function
def parallel_train_step1(data_list):
    sum_loss, predictions, gradients = distributed_strategy.run(train_step1, args=(data_list,))

    predictions = distributed_strategy.experimental_local_results(predictions)
    predictions = tf.concat(predictions, axis=0)

    gradients = distributed_strategy.reduce(tf.distribute.ReduceOp.MEAN, gradients, axis=None)

    return predictions, gradients


@tf.function
def parallel_test_step(images, labels, noise_scales):
    distributed_strategy.run(test_step, args=(images, labels, noise_scales))


@tf.function
def test_step(images, labels, noise_scales):
    predictions = model(images, training=False)
    t_loss = dist.train_loss(labels, predictions, noise_scales)

    test_loss(t_loss)
    test_accuracy(tf.argmax(labels, axis=-1), predictions)


def prepare_checkpoints(model, optimizer, globalstep, avg_vars):
    ckpt = tf.train.Checkpoint(step=globalstep, optimizer=optimizer, model=model, avg_vars=avg_vars)
    manager = tf.train.CheckpointManager(ckpt, train_dir, max_to_keep=1555)

    if load_prev:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"Model restored from {manager.latest_checkpoint}!")
    else:
        print("Initializing new model!")

    # override loaded optimizer options with the default ones
    for option in default_optimizer_config:
        optimizer0._set_hyper(option, default_optimizer_config[option])

    return ckpt, manager


################# program start ##############

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(train_log_dir)
summary_writer.set_as_default()
globalstep = tf.Variable(0, dtype=tf.int64, trainable=False)
tf.summary.experimental.set_step(globalstep)

# warmup to create variables
inputs = [next(ds) for ds in dataset_iters]
predictions, gradients = parallel_train_step1(inputs)
avg_vars = [averager.average(v) for v in model.trainable_variables]

ckpt, ckpt_manager = prepare_checkpoints(model, optimizer, globalstep, avg_vars)

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_accuracy_adv.reset_states()

    test_loss.reset_states()
    test_accuracy.reset_states()

    start_time = time.time()

    for i in range(train_iters):
        inputs = [next(ds) for ds in dataset_iters]
        predictions, gradients = parallel_train_step1(inputs)
        globalstep.assign_add(1)


    for test_images, test_labels, noise_scales in test_ds:
        parallel_test_step(test_images, test_labels, noise_scales)

    tf.summary.scalar('test/loss', test_loss.result())
    tf.summary.scalar('test/accuracy', test_accuracy.result())

    step_time = time.time() - start_time
    start_time = time.time()

    tf.summary.scalar('train/loss', train_loss.result())
    tf.summary.scalar('train/accuracy', train_accuracy.result())
    tf.summary.scalar('train/accuracy_adv', train_accuracy_adv.result())
    tf.summary.histogram('logits', predictions)

    regul = 0
    with tf.name_scope("variables"):
        for var in model.trainable_variables:  # type: tf.Variable
            tf.summary.histogram(var.name, var)
            regul += tf.reduce_sum(tf.abs(var))
    tf.summary.scalar("var_norm", regul)

    with tf.name_scope("gradients"):
        sum_grad = 0
        for grd, var in zip(gradients, model.trainable_variables):
            grad_len = tf.reduce_mean(tf.abs(grd))
            tf.summary.scalar(var.name, grad_len)
            sum_grad += grad_len

    tf.summary.scalar("gradlen", sum_grad)

    summary_writer.flush()

    print(
        f'Epoch {epoch}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}, '
        f'time: {step_time}')

    if epoch % (EPOCHS // total_saved_epochs)==0:
        save_path = ckpt_manager.save()
        print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")
        plot_accuracy(model, test_length)

save_path = ckpt_manager.save()
print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")