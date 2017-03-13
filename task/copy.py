import numpy as np

def generate_data(x_size, y_size, max_seq_size, batch_size, in_seq_min_len, in_seq_max_len):

  x = np.zeros((batch_size, max_seq_size, x_size), dtype=np.float32)
  y = np.zeros((batch_size, max_seq_size, y_size), dtype=np.float32)

  # get random number 1 ~ 20
  #seq_size = in_seq_max_len
  interval = in_seq_max_len - in_seq_min_len + 1
  seq_size = in_seq_min_len + np.random.randint(interval)

  # generate y inputs
  _y = np.random.randint(2, size=(batch_size, seq_size, y_size))
  offset = seq_size + 1 # 1 for delimiter
  y[:, offset:offset + seq_size, :] = _y

  # make x inputs correspoding to y
  x[:, :seq_size, : y_size] = _y
  x[:, 0, y_size] = 1 # mark input start
  x[:, seq_size, (y_size + 1)] = 1 # mark input end
  return x, y
