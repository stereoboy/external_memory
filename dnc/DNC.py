import tensorflow as tf
import tensorflow.contrib.rnn
#from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
import collections

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

_StateTuple = collections.namedtuple("StateTuple", ("c", "h"))

class Controller(RNNCell):

  def __init__(self, x_size, num_units, h_size, w_mem, R, L ):
    """Initialize the basic Controller RNN.

    Args:
      x_size: The dimension of input.
      w_mem: memory word size
      R: number of read heads
      L: number of layers
    """
    self.x_size = x_size
    self._num_units = num_units
    self.w_mem = w_mem
    self.R = R
    self.L = L

  def __call__(self, x, rs, state, scope=None):

    dtype = x.dtype
    batch_size = x.get_shape()[0]
    new_states = []
    new_outs = []
    h = tf.constant( 0.0, [batch_size, self.h_size], dtype=dtype)
    for l in range(self.L):

      s_prev, h_prev = state[l]
      _i, _s_new, _f, _o = rnn_cell._linear([x, rs, h_prev, h], 4*self.h_size, bias=True)

      i = sigmoid(_i)
      f = sigmoid(_f)
      s_new = f*s_prev + i*tanh(_s_new)
      o = sigmoid(_o)
      h_new = o*tanh(s_new)

      new_states.append(_StateTuple(s_new, h_new))
      new_outs.append(h_new))
      h = h_new

    return (new_outs, tuple(new_states))

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return tuple(cell.state_size for cell in self._cells)

def oneplus(x):
  batch_size = x.get_shape()[0]
  one = tf.constant(1.0, dtype=x.dtype)
  return (tf.log(tf.exp(x) + one) + one)

class DNC(object):
  def __init__(self, y_size, batch_size, num_step, Controller, n_mem, dtype):
    self.Controller = Controller
    self.y_size = y_size
    self.w_mem = self.Controller.w_mem
    self.R = self.Controller.R
    self.batch_size = batch_size
    self.n_mem = n_mem

    self.memory = tf.constant(0.0, [batch_size, self.n_mem, self.w_mem], dtype)

    self.xi_split_list = [self.w_mem*self.R , self.R, self.w_mem, 1, self.w_mem, self.w_mem, self.R, 1, 1, self.R*3]


  def __call__(self, x):
    dtype = x.dtype
    size = self.Controller.output_size
    xi_size = sum(self.xi_split_list)
    # unroll LSTM RNN
    outputs = []
    state = self.Controller.zero_state(batch_size, dtype=dtype)
    rs = None # set initial values

    for t in xrange(num_step):
      if t > 0:
        tf.get_variable_scope().reuse_variables()
      cell_out, state = self.Controller(x[:, t, :], rs, state)

      nu = rnn_cell._linear(cell_out, y_size, bias=False)

      xi = rnn_cell._linear(cell_out, xi_size, bias=False)
      k_rs, beta_rs, k_w, beta_w, _e, _v, _fs, _g_a, _g_w, _pis = tf.split(value=xi, self.xi_split_list,  axis=1)

      k_rs = tf.reshape(k_rs, [-1, self.R, self.w_mem])
      beta_rs = tf.reshape(beta_rs, [-1, self.R, 1])
      _fs = tf.reshape(_fs, [-1, self.R, 1])
      _pis = tf.reshape(_pis, [-1, self.R, 3])

      rs = #???
      y = nu + rnn_cell._linear(rs, y_size, bias=False)
      outputs.append(cell_out)

    output = tf.reshape(tf.concat(outputs, 1), [-1, size])

