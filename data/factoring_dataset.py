import itertools
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from Crypto.Util import number  # install PyCryptodome

from config import dist, batch_size, n_classes
from utils import one_hot

test_set = set()
test_lists = {}
test_set_size = 1000
from numpy.random import Generator, PCG64


def to_base(num, b, l=1):
    assert num >= 0
    ans = []
    while num:
        ans.append(num % b)
        num //= b
    while len(ans) < l:
        ans.append(0)
    return ans


def rand_pair(length, variable_length, for_training):
    noise_scale = random.uniform(0, 1)  # sample the nose level uniformly
    inp, res = rand_data_pair(length, variable_length, noise_scale, for_training)
    return inp, res, noise_scale


def rand_batch(length, variable_length, noise_scale, batch_size, for_training):
    inputs = []
    outputs = []

    for i in range(batch_size):
        inp, outp = rand_data_pair(length, variable_length, noise_scale, for_training)
        inputs.append(inp)
        outputs.append(outp)

    return np.array(inputs), np.array(outputs)


def gen_test_set(bit_length):
    rng = Generator(PCG64(seed=1))  # create seeded generator to ensure that test set is always the same
    prime_set = set()
    for n in range(test_set_size):
        primeNum1 = int(number.getPrime(bit_length, rng.bytes))
        primeNum2 = int(number.getPrime(bit_length, rng.bytes))
        if primeNum1.bit_length() <= bit_length and primeNum2.bit_length() <= bit_length:
            prime_set.add((primeNum1, primeNum2))

    test_set.update(prime_set)
    test_lists[bit_length] = list(prime_set)


def get_rand_pair(bitlength, for_training):
    if bitlength not in test_lists:
        gen_test_set(bitlength)

    if for_training:
        found = False
        for iters in range(100):
            d1n = random.getrandbits(bitlength) | 1
            d2n = random.getrandbits(bitlength) | 1
            d1 = [int(one_bit) for one_bit in reversed(bin(d1n)[2:])]
            d2 = [int(one_bit) for one_bit in reversed(bin(d2n)[2:])]
            d1 += [0] * (bitlength - len(d1))
            d2 += [0] * (bitlength - len(d2))

            if (d1n, d2n) not in test_set and (d2n, d1n) not in test_set:
                found = True
                break
        if not found: print("Error generating data not in test set")
    else:
        prime_list = test_lists[bitlength]
        d1n, d2n = random.choice(prime_list)
        d1 = to_base(d1n, 2, bitlength)
        d2 = to_base(d2n, 2, bitlength)

    return d1, d1n, d2, d2n


def rand_data_pair(length, variable_length, noise_scale, for_training):
    length = int(length)  # use python long ints
    if length < 4:
        return [], []
    length = int(length)  # use python long ints
    if length < 4: return [], []
    base = 2
    k = length // 2
    assert k > 0
    while True:
        if variable_length:
            k1 = random.randint((k + 1) // 2, k)
            k2 = k1  # random.randint((k + 1) // 2, k)
        else:
            k1 = k2 = k
        d1, d1n, d2, d2n = get_rand_pair(k1, for_training)
        if d2n > 0 and d1n > 0: break

    d1 += [0] * (k - len(d1))  # zero pad to full length
    d2 += [0] * (k - len(d2))

    mul = to_base(d1n * d2n, base, k * 2)

    both_nums = d1 + d2
    both_nums_hot_clean = one_hot(both_nums, n_classes)
    both_nums_hot = dist.random_at_t(both_nums_hot_clean, length, noise_scale)

    mul_num_hot = one_hot(mul, 2)
    inp = construct_input(both_nums_hot, mul_num_hot, noise_scale, length)
    res = both_nums_hot_clean
    assert len(res) == length
    assert len(inp) == length
    return inp, res



def construct_input(both_nums_hot, mul_num_hot, noise_scale, length, batch_size=0):
    k = length // 2
    inp = np.concatenate([both_nums_hot, mul_num_hot], axis=-1)
    indicator = one_hot([0] * k + [1] * k, 2)
    t_emb = np.zeros([length, 2]) + [np.sin(np.pi * noise_scale), np.cos(np.pi * noise_scale)]
    # t_emb = tf.zeros([length, 2])
    if batch_size > 0:
        indicator = np.tile(np.expand_dims(indicator, 0), [batch_size, 1, 1])
        t_emb = np.tile(np.expand_dims(t_emb, 0), [batch_size, 1, 1])
    inp = np.concatenate([inp, indicator, t_emb], axis=-1)
    return inp


# update the part of inbput with the new x value
def set_input_x_tf(old_data, new_x):
    new_data = tf.concat([new_x, old_data[:, :, n_classes:]], axis=-1)
    return new_data


def extract_x(input):
    x = input[:, :, 0:n_classes]
    return x


def extract_mulnum(input):
    x = input[:, :, n_classes:n_classes + 2]
    return x


def decode_num(num_list):
    num = 0
    mult = 1
    for c in num_list:
        c = int(c)  # to use long python ints
        # if c==0: break
        if not c in range(0, 2): raise Exception("invalid number")
        num += (c - 0) * mult
        mult *= 2

    return num


def factoring_data_generator(length, variable_length=False, for_training=True):
    while True:
        yield rand_pair(length, variable_length, for_training)


def create_factoring_dataset(length, variable_length=False, for_training=True, dataset_size=1000000):
    assert length % 2 == 0

    # Dataset from generator, slower
    divdrop_dataset = tf.data.Dataset.from_generator(factoring_data_generator,
                                                     args=[length, variable_length, for_training],
                                                     output_signature=(
                                                         tf.TensorSpec(shape=(length, 6 + n_classes), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(length, n_classes), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(), dtype=tf.float32)
                                                     ))

    # Dataset from file, faster but takes quite a time to preprocess
    # if for_training:
    #     dataset_path = f"/host-dir/data/integer_factoring/train/len_{length}_{dataset_size}"
    # else:
    #     dataset_path = f"/host-dir/data/integer_factoring/test/len_{length}_{dataset_size}"
    #
    # if not Path(dataset_path).exists():
    #     # Generate dataset and save it to files
    #     divdrop_dataset = tf.data.Dataset.from_generator(div_noise_generator,
    #                                                      args=[length, variable_length, for_training],
    #                                                      output_signature=(
    #                                                          tf.TensorSpec(shape=(length, 6 + n_classes),
    #                                                                        dtype=tf.float32),
    #                                                          tf.TensorSpec(shape=(length, n_classes), dtype=tf.float32),
    #                                                          tf.TensorSpec(shape=(), dtype=tf.float32)
    #                                                      ))
    #     divdrop_dataset = divdrop_dataset.take(dataset_size)
    #     tf.data.experimental.save(divdrop_dataset, dataset_path)
    #
    # # Load the dataset (this avoids from storing dataset in computational graph)
    # divdrop_dataset = tf.data.experimental.load(dataset_path)
    # divdrop_dataset = divdrop_dataset.repeat()
    # divdrop_dataset = divdrop_dataset.shuffle(10000)

    return divdrop_dataset

