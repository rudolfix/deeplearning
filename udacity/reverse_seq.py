# ---
# Problem 3
# ---------
#
# (difficult!)
#
# Write a sequence-to-sequence LSTM which mirrors all the words in a sentence. For example, if your input is:
#
#     the quick brown fox
#
# the model should attempt to output:
#
#     eht kciuq nworb xof
#
# Refer to the lecture on how to put together a sequence-to-sequence model, as well as [this article](http://arxiv.org/abs/1409.3215) for best practices.
#
# ---

import os
import sys
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
import urllib3
import time
import math
from batch_generator import BatchGenerator
from reverse_seq_model import ReverseSeqModel

url = 'http://mattmahoney.net/dc/'
BATCH_SIZE = 256
MIN_CHARS_IN_BATCH = 64
MAX_CHARS_IN_BATCH = 64
STEPS_PER_CHECKPOINT = 50
TRAIN_DIR = 'reverse_seq_train'
DROPOUT_PROB = 0.8
RUN_NAME_SUFFIX = ''


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()


def run_data_directory(num_layers, units_per_layer):
    run_name = "run-%i-%i-%i-%f-%s" % (num_layers, units_per_layer, BATCH_SIZE, DROPOUT_PROB, RUN_NAME_SUFFIX)
    dir = os.path.join(TRAIN_DIR, run_name)
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return  dir


def create_model(session, num_layers, units_per_layer, forward_only):
    model = ReverseSeqModel(units_per_layer, num_layers,
                            BatchGenerator.VOCABULARY_SIZE,
                            MAX_CHARS_IN_BATCH, 5, 0.1, 0.97,
                            forward_only=forward_only)
    run_data_dir = run_data_directory(num_layers, units_per_layer)
    model.summ_writer = tf.train.SummaryWriter(run_data_dir, session.graph_def, flush_secs=1)
    ckpt = tf.train.get_checkpoint_state(run_data_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train(num_layers, units_per_layer):
    print('download and read data')
    filename = maybe_download('text8.zip', 31344016)

    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (num_layers, units_per_layer))
        model = create_model(sess, num_layers, units_per_layer, False)

        # Read data
        text = read_data(filename)
        # create datasets
        valid_size = 1000
        valid_text = text[:valid_size]
        train_text = text[valid_size:]
        # train_size = len(train_text)
        # create batch generators
        train_batch = BatchGenerator(train_text, BATCH_SIZE, MIN_CHARS_IN_BATCH, MAX_CHARS_IN_BATCH,
                                     reverse_encoder_input=True)
        validation_batch = BatchGenerator(valid_text, 1, MIN_CHARS_IN_BATCH, MAX_CHARS_IN_BATCH,
                                          reverse_encoder_input=True)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = model.global_step.eval() + 1
        print('starting from step %i' % current_step)
        previous_losses = []
        enc_state = model.initial_enc_state.eval()
        run_data_dir = run_data_directory(num_layers, units_per_layer)
        while True:
            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, decoder_weights = train_batch.next()
            _, step_loss, enc_state = model.step(sess, current_step, encoder_inputs, decoder_inputs, decoder_weights,
                                                 enc_state, DROPOUT_PROB, False)
            step_time += (time.time() - start_time) / STEPS_PER_CHECKPOINT
            loss += step_loss / STEPS_PER_CHECKPOINT
            current_step += 1
            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % STEPS_PER_CHECKPOINT == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d learning rate %.4f step-time %.2f loss %.3f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, loss, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(run_data_dir, 'state')
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on validation set and print their perplexity.
                encoder_inputs, decoder_inputs, decoder_weights = validation_batch.next()
                _, eval_loss, prediction = model.step(sess, current_step - 1, encoder_inputs, decoder_inputs,
                                                      decoder_weights, enc_state[-1:], 1.0, True)
                eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                print("  eval: loss %.3f, perplexity %.2f" % (eval_loss, eval_ppx))
                print(BatchGenerator.batches2string(encoder_inputs))
                print(BatchGenerator.batches2string(decoder_inputs))
                rolled = np.rollaxis(np.asarray([prediction]), 1, 0)
                splitted = np.vsplit(rolled, rolled.shape[0])
                print(BatchGenerator.batches2string([np.squeeze(e,0) for e in splitted]))
                sys.stdout.flush()


def run_test():
    # test batch generation
    print('download and read data')
    filename = maybe_download('text8.zip', 31344016)
    # Read data
    text = read_data(filename)
    # create datasets
    valid_size = 1000
    valid_text = text[:valid_size]
    train_text = text[valid_size:]
    # train_size = len(train_text)
    # create batch generators
    train_batches = BatchGenerator(train_text, BATCH_SIZE, MIN_CHARS_IN_BATCH, MAX_CHARS_IN_BATCH, reverse_encoder_input=True)
    valid_batches = BatchGenerator(valid_text, 1, MIN_CHARS_IN_BATCH, MAX_CHARS_IN_BATCH)

    # print(BatchGenerator.characters(train_batches.next()[0]))
    e_bs, d_bs, dw_bs = train_batches.next()
    print(BatchGenerator.batches2string(e_bs))
    print(BatchGenerator.batches2string(d_bs))
    BatchGenerator.verify_weights(d_bs, dw_bs)
    e_bs, d_bs, dw_bs = train_batches.next()
    print(BatchGenerator.batches2string(e_bs))
    print(BatchGenerator.batches2string(d_bs))
    BatchGenerator.verify_weights(d_bs, dw_bs)
    e_bs, d_bs, dw_bs = valid_batches.next()
    print(BatchGenerator.batches2string(e_bs))
    print(BatchGenerator.batches2string(d_bs))
    BatchGenerator.verify_weights(d_bs, dw_bs)
    e_bs, d_bs, dw_bs = valid_batches.next()
    print(BatchGenerator.batches2string(e_bs))
    print(BatchGenerator.batches2string(d_bs))
    BatchGenerator.verify_weights(d_bs, dw_bs)


def main(_):
    train(3, 64)
    #run_test()


if __name__ == "__main__":
    tf.app.run()