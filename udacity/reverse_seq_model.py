import random
import numpy as np
import tensorflow as tf
from batch_generator import BatchGenerator


class ReverseSeqModel(object):
    def __init__(self, size, num_layers, vocab_size, max_unrollings,
                 max_gradient_norm, learning_rate,
                 learning_rate_decay_factor, use_lstm=True,
                 forward_only=False):
        """Create the model.

        Args:
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          forward_only: if set, we do not construct the backward pass in the model.
        """
        # self.batch_size = batch_size
        self.max_unrollings = max_unrollings
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.placeholder(tf.float32)
        max_unrollings_summ = tf.scalar_summary("max unrollings", max_unrollings)
        learning_rate_summ = tf.scalar_summary("learning rate", self.learning_rate)
        num_layers_summ = tf.scalar_summary("num of layers", num_layers)
        num_nodes_summ = tf.scalar_summary("nodes per layer", size)
        keep_prob_summ = tf.scalar_summary("dropout prob", self.keep_prob, name='keep_prob')
        self.summ_writer = None # instantiated by upper layer

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # Feeds for inputs.
        self.encoder_inputs = []
        for _ in range(max_unrollings):
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[None, BatchGenerator.VOCABULARY_SIZE]))
        self.decoder_inputs = []
        self.decoder_weights = []
        for i in range(max_unrollings+2):
            self.decoder_inputs.append(tf.placeholder(tf.float32, shape=[None, BatchGenerator.VOCABULARY_SIZE],
                                       name='decoder-inputs-%i' % i))
            self.decoder_weights.append(tf.placeholder(tf.float32, shape=[None], name='decoder-weights-%i' % i))
        # self.zero_state = cell.zero_state()
        # zs = cell.zero_state(256, tf.float32) # tf.placeholder(tf.float32, shape=[None, 2 * size])
        self.initial_enc_state = tf.truncated_normal([256, cell.state_size], -0.1, 0.1)
        # run encoder - decoder
        self.outputs, _, self.enc_state = ReverseSeqModel.basic_rnn_seq2seq(self.encoder_inputs,
                                                                           self.decoder_inputs[:max_unrollings + 1],
                                                                           self.initial_enc_state, cell)
        # our targets are decoder inputs shifted by one.
        targets = self.decoder_inputs[1:]
        # compute logits and loss
        output = tf.concat(0, self.outputs)
        softmax_w = tf.Variable(tf.truncated_normal([size, vocab_size], -0.1, 0.1))
        softmax_b = tf.Variable(tf.zeros([vocab_size]))
        # use a name scope to organize nodes in the graph visualizer
        with tf.name_scope("Wx_b") as _:
            logits = tf.matmul(output, softmax_w) + softmax_b
            self.prediction = tf.nn.softmax(logits)
        # flatten logits and targets
        with tf.name_scope("xent") as _:
            losses = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [tf.concat(0, targets)],
                [tf.concat(0, self.decoder_weights[1:])],
                softmax_loss_function=tf.nn.softmax_cross_entropy_with_logits)
            self.loss = tf.reduce_sum(losses) / 256
            loss_summ = tf.scalar_summary("loss", self.loss)

        # Gradients and SGD update operation for training the model.
        if not forward_only:
            with tf.name_scope("train") as _:
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                gradients, v = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.updates = optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step)
                # gradients_hist = tf.histogram_summary("gradients", tf.concat(gradients,0))
            # gradients = tf.gradients(self.loss, params)
            # clipped_gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, max_gradient_norm)
            # self.gradient_norms.append(norm)
            # self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        self.merged = tf.merge_all_summaries()
        self.saver = tf.train.Saver(tf.all_variables())

    @staticmethod
    def basic_rnn_seq2seq(encoder_inputs, decoder_inputs, state, cell, dtype=tf.float32, scope=None):
        with tf.variable_scope(scope or "basic_rnn_seq2seq"):
            _, enc_state = tf.nn.rnn(cell, encoder_inputs, initial_state=state, dtype=dtype)
            outputs, state = tf.nn.seq2seq.rnn_decoder(decoder_inputs, enc_state, cell)
            return outputs, state, enc_state

    @staticmethod
    def tied_rnn_seq2seq(encoder_inputs, decoder_inputs, state, cell, loop_function=None, dtype=tf.float32, scope=None):
        with tf.variable_scope("combined_tied_rnn_seq2seq"):
            scope = scope or "tied_rnn_seq2seq"
            _, enc_state = tf.nn.rnn(cell, encoder_inputs, initial_state=state, dtype=dtype, scope=scope)
            tf.get_variable_scope().reuse_variables()
            outputs, state = tf.nn.seq2seq.rnn_decoder(decoder_inputs, enc_state, cell,
                                                       loop_function=loop_function, scope=scope)
            return outputs, state, enc_state

    def step(self, session, current_step, encoder_inputs, decoder_inputs, decoder_weights, enc_state, keep_prob,
             forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """

        # Input feed: encoder inputs, decoder inputs, as provided.
        input_feed = {}
        for i in range(len(self.encoder_inputs)):
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]
        for i in range(len(self.decoder_inputs)):
            input_feed[self.decoder_inputs[i]] = decoder_inputs[i]
            input_feed[self.decoder_weights[i]] = decoder_weights[i]
            # print('inserted len %i into placeholder %s' % (len(decoder_weights[i]), self.decoder_weights[i].name))
        input_feed[self.initial_enc_state] = enc_state
        input_feed[self.keep_prob] = keep_prob
        # print('feed %i enc inputs %i dec inputs %i dec weights' % (len(encoder_inputs), len(decoder_inputs),
        #                                                           len(decoder_weights)))

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.loss,  # Loss for this batch.
                           self.enc_state] # encoder state to preserve
        else:
            output_feed = [self.loss,  # Loss for this batch.
                           self.prediction  # Output logits.
                           ]
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # save graph stats
            summary_str = self.merged.eval(input_feed)
            self.summ_writer.add_summary(summary_str, current_step)
            return None, outputs[1], outputs[2]  # loss and state
        else:
            return None, outputs[0], outputs[1]  # No gradient norm, loss, prediction.


class ReverseSeqValidationSummaryModel(object):
    def __init__(self, graph):
        with graph.as_default():
            # validation perplexity summary
            self.validation_perp = tf.placeholder(tf.float32)
            validation_perp_summ = tf.scalar_summary('validation perplexity', self.validation_perp)
            self.merged_validation = tf.merge_all_summaries()
            self.graph = graph