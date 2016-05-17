import string
import numpy as np
import random
from itertools import accumulate
import bisect


class BatchGenerator(object):
    # class vars
    PAD = '.'
    EOS = '#'
    GO = '!'
    UNK = '?'
    VOCAB_START = [PAD, EOS, GO, ' ', UNK]
    VOCABULARY_SIZE = len(string.ascii_lowercase) + len(VOCAB_START)  # [a-z] + specials
    FIRST_LETTER = ord(string.ascii_lowercase[0]) - len(VOCAB_START)  # put PAD + EOS + GO at idx 0,1,2 respectively
    WORD_BREAKERS = [' ']

    def __init__(self, text, batch_size, min_unrollings, max_unrollings, reverse_encoder_input = False,
                 random_batch = True, always_min_unrollings=True):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._min_unrollings = min_unrollings
        self._max_unrollings = max_unrollings
        self._word_boundary_idx = self._idx = 0
        self._reverse_encoder_input = reverse_encoder_input
        self._random_batch = random_batch
        self._always_min_unrollings = always_min_unrollings

    def reverse_word(self, decoder_batch, enc_idx, self_idx):
            idx_diff = (self_idx - self._word_boundary_idx) % self._text_size # needs modulo as idx may wrap
            for dec_idx in range(idx_diff - 1): # do not count word breaker
                dec_c = self._text[self._word_boundary_idx]
                decoder_batch[enc_idx - dec_idx, BatchGenerator.char2id(dec_c)] = 1.0
                self._word_boundary_idx = (self._word_boundary_idx + 1) % self._text_size

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        # generate variable size batch, will be padded to _max_unrollings
        unrollings = self._min_unrollings if self._always_min_unrollings \
            else random.randint(self._min_unrollings, self._max_unrollings)
        if self._random_batch:
            # generate batch from random position on the text, corpus seems to be not mixed well
            self._word_boundary_idx = self._idx = random.randint(0, self._text_size - 1)
        # note that zero is PAD idx, so no need to pad explicitly
        encoder_batch = np.zeros(shape=(self._max_unrollings, BatchGenerator.VOCABULARY_SIZE), dtype=np.float)
        decoder_batch = np.zeros(shape=(self._max_unrollings + 2, BatchGenerator.VOCABULARY_SIZE), dtype=np.float)
        decoder_batch[0, BatchGenerator.char2id(BatchGenerator.GO)] = 1.0
        decoder_batch[unrollings+1, BatchGenerator.char2id(BatchGenerator.EOS)] = 1.0
        # decoder weights == 0 for pad characters
        decoder_weights = np.ones(shape=(self._max_unrollings + 2), dtype=np.float)
        decoder_weights[unrollings+2:] = 0

        for enc_idx in range(unrollings):
            c = self._text[self._idx]
            encoder_batch[enc_idx, BatchGenerator.char2id(c)] = 1.0
            self._idx = (self._idx + 1) % self._text_size
            # if word boundary then generate decoder_batch fragment
            if c in BatchGenerator.WORD_BREAKERS:
                decoder_batch[enc_idx + 1, BatchGenerator.char2id(c)] = 1.0 # put word breaker + 1 for GO char
                self.reverse_word(decoder_batch, enc_idx, self._idx)
                self._word_boundary_idx = (self._word_boundary_idx + 1) % self._text_size # skip word breaker char
        # reverse remainder
        # todo: when corpus boundary is crossed amd self._idx == self._word_boundary_idx this is not fired! FIX
        if self._idx != self._word_boundary_idx:
            self.reverse_word(decoder_batch, enc_idx + 1, (self._idx + 1) % self._text_size)

        # no need for additional padding
        # for idx in range(unrollings, self._max_unrollings):
        #    encoder_batch[idx, BatchGenerator.char2id(BatchGenerator.PAD)] = 1.0
        #    decoder_batch[idx + 1, BatchGenerator.char2id(BatchGenerator.PAD)] = 1.0

        return encoder_batch, decoder_batch, decoder_weights

    def next(self):
        t_encoder_batches = []
        t_decoder_batches = []
        t_decoder_weights = []
        for step in range(self._batch_size):
            e_b, d_b, dw_b = self._next_batch()
            t_encoder_batches.append(e_b if not self._reverse_encoder_input else np.flipud(e_b))
            t_decoder_batches.append(d_b)
            t_decoder_weights.append(dw_b)
        # map from the list of batches to the list of unrollings, where the batch is a "column"
        encoder_batches = np.rollaxis(np.asarray(t_encoder_batches), 1, 0)
        decoder_batches = np.rollaxis(np.asarray(t_decoder_batches), 1, 0)
        decoder_weights = np.rollaxis(np.asarray(t_decoder_weights), 1, 0)

        return [np.squeeze(a, 0) for a in np.vsplit(encoder_batches, encoder_batches.shape[0])], \
            [np.squeeze(a, 0) for a in np.vsplit(decoder_batches, decoder_batches.shape[0])], \
            [np.squeeze(a, 0) for a in np.vsplit(decoder_weights, decoder_weights.shape[0])]

    @staticmethod
    def characters(probabilities):
        return [BatchGenerator.id2char(c) for c in np.argmax(probabilities, 1)]

    @staticmethod
    def batches2string(batches):
        s = [''] * batches[0].shape[0]
        for b in batches:
            s = [''.join(x) for x in zip(s, BatchGenerator.characters(b))]
        return s
        # return [''.join(BatchGenerator.characters(b)) for b in batches]

    @staticmethod
    def verify_weights(decoder_batches, decoder_weights):
        for bidx in range(len(decoder_batches)):
            for chidx in range(len(decoder_batches[bidx])):
                ch = BatchGenerator.id2char(np.argmax(decoder_batches[bidx][chidx]))
                w = decoder_weights[bidx][chidx]
                if ch == BatchGenerator.PAD and w != 0 or ch != BatchGenerator.PAD and w == 0:
                    raise Exception('for batch %i idx %i w is %f and ch is %c' % (bidx, chidx, w, ch))

    @staticmethod
    def char2id(char):
        if char in string.ascii_lowercase:
            return ord(char) - BatchGenerator.FIRST_LETTER
        elif char in BatchGenerator.VOCAB_START:
            return BatchGenerator.VOCAB_START.index(char)
        else:
            print('Unexpected character: %s' % char)
            return BatchGenerator.VOCAB_START.index(BatchGenerator.UNK)

    @staticmethod
    def id2char(dictid):
        if dictid >= len(BatchGenerator.VOCAB_START):
            return chr(dictid + BatchGenerator.FIRST_LETTER)
        else:
            return BatchGenerator.VOCAB_START[dictid]

    @staticmethod
    def logprob(predictions, labels):
        predictions[predictions < 1e-10] = 1e-10
        return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


class RandomWordsBatchGenerator(object):
    # http://norvig.com/mayzner.html
    word_lengths_frequencies = [2.998, 17.651, 20.511, 14.787, 10.700, 8.388, 7.939, 5.943, 4.437, 3.076, 1.761, 0.958]
    wlf_cumul = list(accumulate(word_lengths_frequencies))
    wlf_cumul.append(100.001)

    def __init__(self, batch_size, min_unrollings, max_unrollings, reverse_encoder_input=False):
        self._batch_size = batch_size
        self._min_unrollings = min_unrollings
        self._max_unrollings = max_unrollings
        self._reverse_encoder_input = reverse_encoder_input

    @staticmethod
    def gen_word():
        # we check where random float lands on a line where word freq probs are represented as lengths
        wf_prob = random.random() * 100.0
        w_len = bisect.bisect(RandomWordsBatchGenerator.wlf_cumul, wf_prob) + 1
        sentence = []
        for _ in range(w_len):
            sentence.append(chr(random.randint(ord('a'), ord('z'))))
        return ''.join(sentence)

    def next(self):
        # generate string of random letters that make words that approximate english in length
        sentence = ''
        while len(sentence) < self._max_unrollings * self._batch_size:
            sentence += self.gen_word() + ' '

        sentence = sentence[:self._max_unrollings * self._batch_size]
        batch_generator = BatchGenerator(sentence, self._batch_size, self._min_unrollings, self._max_unrollings,
                                         reverse_encoder_input=self._reverse_encoder_input, random_batch=False)
        return batch_generator.next()


class ReverseStringBatchGenerator(object):
    def __init__(self, batch_size, min_unrollings, max_unrollings, reverse_encoder_input=False):
        self._batch_size = batch_size
        self._min_unrollings = min_unrollings
        self._max_unrollings = max_unrollings
        self._reverse_encoder_input = reverse_encoder_input

    @staticmethod
    def gen_word(strlen):
        sentence = []
        for _ in range(strlen):
            sentence.append(chr(random.randint(ord('a'), ord('z'))))
        return ''.join(sentence)

    def next(self):
        # generate string of random letters
        sentence = self.gen_word(self._max_unrollings * (self._batch_size+1))
        batch_generator = BatchGenerator(sentence, self._batch_size, self._min_unrollings, self._max_unrollings,
                                         reverse_encoder_input=self._reverse_encoder_input, random_batch=False,
                                         always_min_unrollings=True)
        return batch_generator.next()