import tensorflow as tf
#import tensorflow.contrib.rnn.python.ops.core_rnn_cell as rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl as rnn_cell

#from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
import collections

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.ops import array_ops

_StateTuple = collections.namedtuple("StateTuple", ("c", "h"))


class Controller(RNNCell):

  def __init__(self, x_size, h_size, w_mem, num_layers ):
    """Initialize the basic Controller RNN.

    Args:
      x_size: The dimension of input.
      w_mem: memory word size
      R: number of read heads
      L: number of layers
    """
    self.x_size = x_size
    self.h_size = h_size
    self.w_mem = w_mem
    self.num_layers = num_layers

  def __call__(self, x, reads, state, scope=None):

    dtype = x.dtype
    batch_size = x.get_shape()[0]
    new_states = []
    new_outs = []
    h = tf.concat(values=[x] + reads, axis=1)
    with tf.variable_scope("Controller"):
      for l in range(self.num_layers):

        if self.num_layers > 1:
          s_prev, h_prev = state[l]
        else:
          print state
          s_prev, h_prev = state

        h = tf.concat(values=[x] + reads, axis=1)
        with tf.variable_scope("cell_%d" % l):
          ret = rnn_cell._linear([h_prev, h], 4*self.h_size, bias=True)
        _i, _s_new, _f, _o = array_ops.split(value=ret, num_or_size_splits=4, axis=1)

        i = sigmoid(_i)
        f = sigmoid(_f)
        s_new = f*s_prev + i*tanh(_s_new)
        o = sigmoid(_o)
        h_new = o*tanh(s_new)
        
        if self.num_layers > 1:
          new_outs.append(h_new)
          new_states.append(_StateTuple(s_new, h_new))
        else:
          return (h_new,  _StateTuple(s_new, h_new))
        h = h_new

    return (new_outs, tuple(new_states))

  @property
  def output_size(self):
    return self.h_size

  @property
  def state_size(self):
    if self.num_layers > 1:
      return tuple( (self.h_size, self.h_size) for _ in xrange(self.num_layers))
    else:
      return (self.h_size, self.h_size)

def oneplus(x):
  return (tf.nn.softplus(x) + 1)

def circular_convolution(w, s):
  
  # build NxN matrix
  def shift(s, i):
    n = s.get_shape()[1]
    #print s, n, i
    #print [i, int(n-i)]
    if i > 0 and i < n - 1:
      left, right = array_ops.split(value=s, num_or_size_splits=[i, int(n-i)], axis=1)
      s_ = tf.concat([right, left], axis=1)
      return s_
    else:
      return s

  if w.get_shape() != s.get_shape():
    raise ValueError("w == s")
  
  _S = []

  n = w.get_shape()[1]
  for i in xrange(n):
    s_ = shift(s, i)
    _S.append(s_)
  S = tf.stack(_S, axis=1) 
  w_ = tf.expand_dims(w, axis=2)
  w_ = tf.matmul(S, w_)
  w = tf.squeeze(w_, axis=2)
  return w

def focus_by_context(beta, k, memory):
  
  def similarity(u, v):
    #norm = tf.norm(u, axis=2, keep_dims=True)*tf.norm(v, axis=2, keep_dims=True)
    u = u/tf.norm(u, axis=2, keep_dims=True)
    v = v/tf.norm(v, axis=2, keep_dims=True)
    u = tf.Print(u, [u], message="u:")
    v = tf.Print(v, [v], message="v:")
    #norm = tf.Print(norm, [norm], message="norm:")
    ret = tf.matmul(u, v, transpose_b=True)
    return ret

  k_ = tf.expand_dims(k, axis=1) # expand dim batch x 1 x w_mem

  k_ = tf.Print(k_, [k_], message="k_:")
  _w = similarity(memory, k_) # batch x n_mem x 1

  _w = tf.Print(_w, [_w], message="_w:")
  _w = tf.squeeze(_w, axis=[2]) # batch x n_mem
  _w = beta*_w
  w = tf.nn.softmax(_w)
  return w

def focus_by_location(w, w_prev, s, g, gamma):
  w_ = g*w + (1-g)*w_prev
  w_ = circular_convolution(w_, s)
  w_ = tf.pow(w_, gamma)
  w_ = w_/tf.reduce_sum(w_, axis=1, keep_dims=True)
  return w_

def addressing(heads, memory, ws_prev):
  ws = []
  for w_prev, head in zip(ws_prev, heads):
    k = head['k']
    beta =  head['beta']
    g = head['g']
    s = head['s']
    gamma = head['gamma']

    #gamma = tf.Print(gamma, [gamma])
    w = focus_by_context(beta, k, memory)
    w = focus_by_location(w, w_prev, s, g, gamma)
    ws.append(w)
  return ws

def process_read(memory, ws):
  reads = []
  for w in ws:
    w = tf.expand_dims(w, axis=1)
    _read = tf.matmul(w, memory)
    read = tf.squeeze(_read, axis=[1])
    reads.append(read)
  return reads

def update_memory(memory, add_vec, erase_vec, write_ws):
  for w in write_ws:
    w_ = tf.expand_dims(w, axis=2) # batch x n_mem x 1
    a = tf.expand_dims(add_vec, axis=1) # batch x 1 x w_mem
    e = tf.expand_dims(erase_vec, axis=1) # batch x 1 x w_mem

    A = tf.matmul(w_, a) # batch x n_mem x w_mem
    E = tf.matmul(w_, e) # batch x n_mem x w_mem
    memory = memory* (1 - E) + A

  return memory

def init_weight(batch_size, n, dtype):
  
  _ret = tf.constant(1.0, dtype=dtype, shape=[batch_size, n])
  ret = _ret/tf.reduce_sum(_ret, axis=1, keep_dims=True)
  return ret

def init_vector(batch_size, dim, dtype):

  ret = tf.constant(0.0, dtype=dtype, shape=[batch_size, dim])
  return ret

def debug_scope():
  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="/*")

  print "============================"
  for item in var_list:
    print item.name

class NTMCell(object):
  def __init__(self, y_size, batch_size, num_step, Controller, R, W, n_mem, dtype):
    self.Controller = Controller
    self.y_size = y_size
    self.w_mem = self.Controller.w_mem
    self.R = R
    self.W = W 
    self.batch_size = batch_size
    self.n_mem = n_mem

    self.head_split_list = [self.w_mem, 1, 1, self.n_mem, 1] # for k, beta, g, s, gamma
    head_size = sum(self.head_split_list)
    self.xi_split_list = [head_size]*(R + W) + [self.w_mem] + [self.w_mem]


  def __call__(self, x):
    dtype = x.dtype
    batch_size = x.get_shape()[0]
    num_step = x.get_shape()[1]
    xi_size = sum(self.xi_split_list)
    # unroll LSTM RNN
    outputs = []

    with tf.variable_scope("NTM"):
      self.memory = tf.constant(0.001, dtype=dtype, shape=[batch_size, self.n_mem, self.w_mem])
      state = self.Controller.zero_state(batch_size, dtype=dtype)
      reads = [init_vector(batch_size, self.w_mem, dtype) for _ in xrange(self.R)]
      read_ws = [init_weight(batch_size, self.n_mem, dtype) for _ in xrange(self.R)]
      write_ws = [init_weight(batch_size, self.n_mem, dtype) for _ in xrange(self.W)]
      for t in xrange(num_step):
        print "step:", t
        if t > 0:
          tf.get_variable_scope().reuse_variables()
        cell_out, state = self.Controller(x[:, t, :], reads, state)

        with tf.variable_scope("Nu"):
          nu = rnn_cell._linear(cell_out, self.y_size, bias=False)
        
        with tf.variable_scope("Xi"):
          xi = rnn_cell._linear(cell_out, xi_size, bias=False)
        
        _head_params = tf.split(value=xi, num_or_size_splits=self.xi_split_list, axis=1)

        # extract add_vec, erase_vec
        _add_vec, _erase_vec = _head_params[-2:]
        add_vec = tf.sigmoid(_add_vec)
        erase_vec = tf.sigmoid(_erase_vec)
        # extract head parameters from controller outputs
        read_heads = []
        write_heads = []
        for i, params in enumerate(_head_params[:-2]):
          head = {}
          _k, _beta, _g, _s, _gamma = tf.split(value=params, num_or_size_splits=self.head_split_list, axis=1)
          head['k'] = _k
          head['beta'] = oneplus(_beta)
          head['g'] = tf.sigmoid(_g)
          head['s'] = tf.nn.softmax(_s)
          head['gamma'] = oneplus(_gamma)
          if i < self.R:
            read_heads.append(head)
          else:
            write_heads.append(head)

        read_ws = addressing(read_heads, self.memory, read_ws)
        write_ws = addressing(write_heads, self.memory, write_ws)

        reads = process_read(self.memory, read_ws)
        self.memory = update_memory(self.memory, add_vec, erase_vec, write_ws)

        with tf.variable_scope("Out"):
          y = nu + rnn_cell._linear(reads, self.y_size, bias=False)
        outputs.append(y)
        
        if t==0:
          debug_out = read_ws[0]

        debug_scope()


    #debug_out = xi
    #output = tf.reshape(tf.concat(values=outputs, axis=1), [-1, self.y_size])
    output = tf.stack(values=outputs, axis=1)

    return output, debug_out

