import tensorflow as tf

def oneplus(x):
  return (tf.nn.softplus(x) + 1)

def print_debug(tensor, message):
  tensor = tf.Print(tensor, [tensor], message=message)
  return tensor 
