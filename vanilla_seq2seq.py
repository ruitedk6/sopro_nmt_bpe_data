# coding: utf-8

# Dependencies
import numpy as np
import tensorflow as tf
from preproc_pipe import *

#tf.reset_default_graph()
sess = tf.InteractiveSession()

import numpy as np

def make_batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
]



# read the prepare the data here

batch_size = 64

iterator, word2index, index2word = prepare_data('test1.bpe.eng', 'test.bpe.ara', 50, 128)

print('head of the batch:')

tgt_list, src_list = [], []
tgt_vocab, src_vocab = set(), set()

# switch src and tgt between Arabic and English
for (tgt, src) in iterator:
    for w in src:
        tgt_vocab.add(w)
        
    for w in tgt:
        src_vocab.add(w)
    
    tgt_list.append(src)
    src_list.append(tgt)
    
# partition lists into chuncks of size batch_size
def make_chunks(dataset, n):
    # accepts a list and an integer (n), yields n chuncks of the list
    for i in range(0, len(dataset), n):
        yield dataset[i:i + n]
        
chunked_src = make_chunks(src_list, batch_size)
chunked_tgt = make_chunks(tgt_list, batch_size)
print(len(src_vocab), len(tgt_vocab))


# In[4]:

PAD = 0
EOS = 1

vocab_size = len(src_vocab) + 2
input_embedding_size = 64 # character length 

encoder_hidden_units = 64
decoder_hidden_units = encoder_hidden_units * 2



# input placeholders 
# this is the batch! a tensor of tf.int32
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')

# contains the length for each sequence in a batch, we will pad so have same lenght
# if you don't want to pad, check out dynamic memory networks to input variable length sequence 
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_input_length')

# target sequence for pairs in a batch
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


# In[6]:

# embeddings: a tensor of tf.float32
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

# this looks interesting!
# tf.nn.embedding_lookup takes single tensor representing the complete embeddings,
# and  a tensor with type int32 containing the ids to be looked up in embeddings
# returns a tensor with the same type as the tensors in embeddings
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)


# In[7]:

# define encoder's forward and backward LSTM
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

# define fw and bw cells 
with tf.variable_scope('forward'):
    encoder_fw_cell = LSTMCell(encoder_hidden_units)
with tf.variable_scope('backward'):
    encoder_bw_cell = LSTMCell(encoder_hidden_units)
    
# initilize BRNN
((encoder_fw_outputs,
 encoder_bw_outputs), 
(encoder_fw_final_state,
encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw_cell,
                                    cell_bw=encoder_bw_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
                            )

# bidirectional step
# concatenates tensors along one dimension
# the resulting tensor of this concatenation has shape (?, 40)
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 1)

# h: hidden state, c: cell state
encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

# create an LSTMStateTuple object for the encoder's final state
encoder_final_state = LSTMStateTuple(
    c = encoder_final_state_c,
    h = encoder_final_state_h
)


# decoder cell
decoder_cell = LSTMCell(decoder_hidden_units)

# tf.unstack returns the shapes of the tensor: max sequence length and batch size
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

# length of the target sequence (in training)
# NOTE: I am not sure why Siraj put +3 here!!!
decoder_lengths = encoder_inputs_length # + 3

# Added by Badr
# here, we should have something similar to encoder_inputs_embedding 
decoder_targets_embedded = tf.nn.embedding_lookup(embeddings, decoder_targets)

#decoder_targets_embedded_ta = tf.TensorArray(dtype=tf.float32, size=encoder_inputs_length[0])
#decoder_targets_embedded_ta = decoder_targets_embedded_ta.unstack(decoder_targets_embedded)


tf.shape(encoder_inputs), tf.unstack(tf.shape(encoder_inputs)), decoder_lengths, (encoder_max_time, batch_size)


# outputs projection
# define our weights and biases 
# these weights correspond to the input matrix  
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)


# create padded inputs for the decoder from the word embeddings

# were telling the program to test a condition, and trigger an error if the condition is false.
assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

# retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

# manually specifying loop function through time - to get initial cell state and input to RNN
# normally we'd just use dynamic_rnn, but lets get detailed here with raw_rnn

# we define and return these values, no operations occur here
def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    #end of sentence
    initial_input = eos_step_embedded
    # last time steps cell state
    initial_cell_state = encoder_final_state
    # none
    initial_cell_output = None
    # none
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)


# attention mechanism --choose which previously generated token to pass as input in the next timestep
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    
    def get_next_input():
        # dot product between previous ouput and weights, then + biases
        
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        # Logits simply means that the function operates on the unscaled output of 
        # earlier layers and that the relative scale to understand the units is linear. 
        # It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities 
        # (you might have an input of 5).
        # prediction value at current time step
        
        # Returns the index with the largest value across axes of a tensor.
        # This is attention!! Nope it is not
        # This line should not be applied during training 
        # it would be possible to use this line during inference
        # instead, the next_input should be the item in the ground truth data, not the predicted
        # for teacher forcing (also known as MLE)
        prediction = tf.argmax(output_logits, axis=1)

        # embed prediction for the next input
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input 
    
    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended
   
    # Computes the "logical and" of elements across dimensions of a tensor.
    finished = tf.reduce_all(elements_finished) # -> boolean scalar
    
    # Return either fn1() or fn2() based on the boolean predicate pred.
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    #input = tf.cond(finished, lambda: pad_step_embedded, lambda: decoder_targets_embedded_ta.read(time))
    
    # set previous to current
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished, 
            input,
            state,
            output,
            loop_state)


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

#Creates an RNN specified by RNNCell cell and loop function loop_fn.
#This function is a more primitive version of dynamic_rnn that provides more direct access to the 
#inputs each iteration. It also provides more control over when to start and finish reading the sequence, 
#and what to emit for the output.
#ta = tensor array
decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

# to convert output to human readable prediction
# we will reshape output tensor

# Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
# reduces dimensionality
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
# flettened output tensor
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
# pass flattened tensor through decoder
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
# prediction vals
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))


# final prediction
decoder_prediction = tf.argmax(decoder_logits, 2)


# cross entropy loss
# one hot encode the target values so we don't rank just differentiate
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

# loss function
loss = tf.reduce_mean(stepwise_cross_entropy)
# train it 
train_op = tf.train.AdamOptimizer().minimize(loss)


def next_feed(batch):
    (src_batch, tgt_batch) = batch
    encoder_inputs_batch, encoder_input_lengths_ = make_batch(src_batch)
    decoder_targets_, _ = make_batch(src_batch)

    return {
        encoder_inputs: encoder_inputs_batch,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }


loss_track = []


# In[34]:

batches_in_epoch = len(src_list) // batch_size

epochs = 100

sess.run(tf.global_variables_initializer())

try:
    for epoch in range(epochs):
        batch_size = 64
        chunked_src = make_chunks(src_list, batch_size)
        chunked_tgt = make_chunks(tgt_list, batch_size)
        for batch_num, batch in enumerate(zip(chunked_src, chunked_tgt)):
            fd = next_feed(batch)
            #print(fd[encoder_inputs].shape)
            #print(fd[decoder_targets].shape)
            #print(fd[encoder_inputs])
            #print(fd[decoder_targets])
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)


            if batch_num == 0 or batch_num % batches_in_epoch == 0:
                print('epcoh {},  batch {}'.format(epoch, batch_num))
                print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format(inp))
                    print('    predicted > {}'.format(pred))
                    if i >= 2:
                        break
                print()

except KeyboardInterrupt:
    print('training interrupted')
