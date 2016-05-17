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
import fileinput
import numpy as np
import tensorflow as tf
import zipfile
import urllib3
import time
import math
from batch_generator import BatchGenerator, RandomWordsBatchGenerator, ReverseStringBatchGenerator
from reverse_seq_model import ReverseSeqModel, ReverseSeqValidationSummaryModel
from enum import Enum


class UseTrainBatchType(Enum):
    use_english_words = 1
    use_random_train_words = 2,
    use_random_string = 3

BATCH_SIZE = 256
MIN_CHARS_IN_BATCH = 32
MAX_CHARS_IN_BATCH = 32
NUM_LAYERS = 4
UNITS_PER_LAYER = 300
DROPOUT_PROB = 0.6
LEARNING_RATE = 0.03
REVERSE_ENCODER_INPUT = False
DECODER_FEED_PREVIOUS = False
USE_LSTM = True
USE_ATTENTION = False
TRAIN_BATCH_TYPE = UseTrainBatchType.use_random_train_words
RUN_NAME_SUFFIX = 'L0.03-NOPAD-RANDOMWORDS-32CHARS'  # additional suffix to describe run names
# L0.03-NOPAD-RANDOMWORDS

GRADIENT_CLIP = 5
LEARNING_RATE_DECAY_RATIO = 0.97
STEPS_PER_CHECKPOINT = 50
TRAIN_DIR = 'reverse_seq_train'


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urlretrieve('http://mattmahoney.net/dc/' + filename, filename)
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
    path = os.path.join(TRAIN_DIR, run_name)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def create_model(session, num_layers, units_per_layer, forward_only, decoder_feed_previous):
    model = ReverseSeqModel(BATCH_SIZE, units_per_layer, num_layers,
                            BatchGenerator.VOCABULARY_SIZE,
                            MAX_CHARS_IN_BATCH, GRADIENT_CLIP, LEARNING_RATE, LEARNING_RATE_DECAY_RATIO,
                            forward_only=forward_only, use_lstm=USE_LSTM, feed_previous=decoder_feed_previous,
                            use_attention=USE_ATTENTION)
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

    with tf.Session(graph=tf.Graph()) as validation_session:
        validation_model = ReverseSeqValidationSummaryModel(validation_session.graph)
        validation_session.run(tf.initialize_all_variables())

        with tf.Session(graph=tf.Graph()) as sess:
            # Create model.
            print("Creating %d layers of %d units." % (num_layers, units_per_layer))
            model = create_model(sess, num_layers, units_per_layer, False, DECODER_FEED_PREVIOUS)

            # Read data
            text = read_data(filename)
            # create datasets
            valid_size = 10000
            valid_text = text[:valid_size]
            train_text = text[valid_size:]
            # create batch generators
            validation_batch = BatchGenerator(valid_text, 1, MIN_CHARS_IN_BATCH, MAX_CHARS_IN_BATCH,
                                              reverse_encoder_input=REVERSE_ENCODER_INPUT)
            if TRAIN_BATCH_TYPE == UseTrainBatchType.use_english_words:
                train_batch = BatchGenerator(train_text, BATCH_SIZE, MIN_CHARS_IN_BATCH, MAX_CHARS_IN_BATCH,
                                             reverse_encoder_input=REVERSE_ENCODER_INPUT)
            elif TRAIN_BATCH_TYPE == UseTrainBatchType.use_random_train_words:
                train_batch = RandomWordsBatchGenerator(BATCH_SIZE, MIN_CHARS_IN_BATCH, MAX_CHARS_IN_BATCH,
                                                        reverse_encoder_input=REVERSE_ENCODER_INPUT)
            else:
                train_batch = ReverseStringBatchGenerator(BATCH_SIZE, MIN_CHARS_IN_BATCH, MAX_CHARS_IN_BATCH,
                                                          reverse_encoder_input=REVERSE_ENCODER_INPUT)
                validation_batch = ReverseStringBatchGenerator(1, MIN_CHARS_IN_BATCH, MAX_CHARS_IN_BATCH,
                                                               reverse_encoder_input=REVERSE_ENCODER_INPUT)


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
                    val_perp = validate_sentence(sess, model, validation_batch, enc_state, current_step)
                    summary_str = validation_model.merged_validation.eval(
                        {validation_model.validation_perp: val_perp if val_perp < 500 else 500 },
                        validation_session)
                    model.summ_writer.add_summary(summary_str, current_step)
                    sys.stdout.flush()
                    # decode_sentence(sess, model, enc_state, current_step)


def decode(num_layers, units_per_layer):
    print('Will decode')
    with tf.Session(graph=tf.Graph()) as sess:
        # Create model.
        print("Creating %d layers of %d units." % (num_layers, units_per_layer))
        model = create_model(sess, num_layers, units_per_layer, False, False)
        current_step = model.global_step.eval() + 1
        enc_state = model.initial_enc_state.eval()
        print('starting from step %i' % current_step)
        # Decode from standard input.
        decode_sentence(sess, model, enc_state, current_step)


def decode_sentence(sess, model, enc_state, current_step):
    print('press ENTER to exit decoder')
    sentence = input('>')
    while sentence:
        # make it exactly max_unrollings and not less than min_unrollings
        sentence = sentence[:MAX_CHARS_IN_BATCH]
        len_s = len(sentence)
        sentence += '.'*(MAX_CHARS_IN_BATCH-len(sentence)) + 'aaa'
        # set MAX_CHARS_IN_BATCH as min_unrollings as we prepared the string already
        validation_batch = BatchGenerator(sentence, 1, len_s, MAX_CHARS_IN_BATCH,
                                          reverse_encoder_input=REVERSE_ENCODER_INPUT, random_batch=False,
                                          always_min_unrollings=True)
        validate_sentence(sess, model, validation_batch, enc_state, current_step)
        sentence = input('>')


def validate_sentence(session, model, validation_batch, encoder_state, current_step):
    encoder_inputs, single_decoder_inputs, decoder_weights = validation_batch.next()
    print(BatchGenerator.batches2string(encoder_inputs))
    print(BatchGenerator.batches2string(single_decoder_inputs))
    # replicate to full batch size so we have multiple results agains the whole state
    encoder_inputs = [np.repeat(x, BATCH_SIZE, axis=0) for x in encoder_inputs]
    decoder_inputs = [np.repeat(x, BATCH_SIZE, axis=0) for x in single_decoder_inputs]
    decoder_weights = [np.repeat(x, BATCH_SIZE, axis=0) for x in decoder_weights]
    # _, eval_loss, prediction = model.step(sess, current_step - 1, encoder_inputs, decoder_inputs,
    #                                      decoder_weights, enc_state[-1:], 1.0, True)
    _, eval_loss, prediction = model.step(session, current_step - 1, encoder_inputs, decoder_inputs,
                                          decoder_weights, encoder_state, 1.0, True)
    # split into 'no of batches' list then average across batches
    reshaped = np.reshape(prediction, (prediction.shape[0] / BATCH_SIZE, BATCH_SIZE, prediction.shape[1]))
    averaged = np.mean(reshaped, axis=1)
    # now roll as in case of single batch
    rolled = np.rollaxis(np.asarray([averaged]), 1, 0)
    splitted = np.vsplit(rolled, rolled.shape[0])
    squeezed = [np.squeeze(e,0) for e in splitted]
    print(BatchGenerator.batches2string(squeezed))
    # compute character to character perplexity
    val_perp = float(np.exp(BatchGenerator.logprob(np.concatenate(squeezed),
                                                   np.concatenate(single_decoder_inputs[1:]))))
    print('--validation perp.: %.2f' % val_perp)
    return val_perp


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
    print('test main batch generator')
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

    print('test random english generator')
    random_batch = RandomWordsBatchGenerator(2, MIN_CHARS_IN_BATCH, MAX_CHARS_IN_BATCH,
                                             reverse_encoder_input=False)
    for _ in range(10):
        e_bs, d_bs, dw_bs = random_batch.next()
        print(BatchGenerator.batches2string(e_bs))
        print(BatchGenerator.batches2string(d_bs))
        BatchGenerator.verify_weights(d_bs, dw_bs)

    print('test random string gen with padding')
    random_str_batch = ReverseStringBatchGenerator(1, 8, 8,
                                             reverse_encoder_input=False)
    e_bs, d_bs, dw_bs = random_str_batch.next()
    print(BatchGenerator.batches2string(e_bs))
    print(BatchGenerator.batches2string(d_bs))
    BatchGenerator.verify_weights(d_bs, dw_bs)
    random_str_batch = ReverseStringBatchGenerator(2, 8, 16,
                                             reverse_encoder_input=False)
    e_bs, d_bs, dw_bs = random_str_batch.next()
    print(BatchGenerator.batches2string(e_bs))
    print(BatchGenerator.batches2string(d_bs))
    BatchGenerator.verify_weights(d_bs, dw_bs)


def main(_):
    if FLAGS.self_test:
        run_test()
    elif FLAGS.decode:
        decode(NUM_LAYERS, UNITS_PER_LAYER)
    elif FLAGS.train:
        train(NUM_LAYERS, UNITS_PER_LAYER)
    else:
        print('set decode or train flags')

tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("train", True,
                            "trains the network if this is set to True.")

FLAGS = tf.app.flags.FLAGS

if __name__ == "__main__":
    tf.app.run()