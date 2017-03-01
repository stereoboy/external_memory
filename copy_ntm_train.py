import numpy as np
import tensorflow as tf
from datetime import datetime, date, time
from ntm.NTM import Controller
from ntm.NTM import NTMCell
import cv2
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("num_layers", "1", "Controller layer size")
tf.flags.DEFINE_integer("x_size", "10", "input dimensions")
tf.flags.DEFINE_integer("h_size", "100", "hidden dimensions for NTM controller")
tf.flags.DEFINE_integer("y_size", "8", "output_dimensions")
tf.flags.DEFINE_integer("N", "128", "number of memory slot")
tf.flags.DEFINE_integer("W", "20", "dimension of memory slot")
tf.flags.DEFINE_integer("Reads", "1", "number of read heads")
tf.flags.DEFINE_integer("Writes", "1", "number of write heads")
tf.flags.DEFINE_integer("batch_size", "16", "batch_size")
tf.flags.DEFINE_integer("in_seq_max_len", "1", "input seq max_size")

tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Momentum Optimizer")
tf.flags.DEFINE_float("momentum", "0.9", "momentum for Momentum Optimizer")
tf.flags.DEFINE_integer("max_epoch", "10", "maximum iterations for training")
tf.flags.DEFINE_integer("max_itrs", "10000", "maximum iterations for training")

tf.flags.DEFINE_string("save_dir", "ntm_checkpoints", "dir for checkpoints")
tf.flags.DEFINE_integer("save_itr", "200", "checkpoint interval")


def generate_data(x_size, y_size, max_seq_size, batch_size):

  x = np.zeros((batch_size, max_seq_size, x_size), dtype=np.float32)
  y = np.zeros((batch_size, max_seq_size, y_size), dtype=np.float32)

  # get random number 1 ~ 20
  seq_size = 1 + np.random.randint(FLAGS.in_seq_max_len)
  
  # generate y inputs
  _y = np.random.randint(2, size=(batch_size, seq_size, y_size))
  offset = seq_size + 1 # 1 for delimiter
  y[:, offset:offset + seq_size, :] = _y
  
  # make x inputs correspoding to y 
  x[:, :seq_size, : y_size] = _y
  x[:, 0, y_size] = 1 # mark input start
  x[:, seq_size, (y_size + 1)] = 1 # mark input end
  return x, y

def get_opt(loss, scope):
  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  print "============================"
  print scope
  for item in var_list:
    print item.name
  print "============================"
  optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
  print "============================"
  grads = optimizer.compute_gradients(loss, var_list=var_list)
  print "============================"
  return optimizer.apply_gradients(grads)

def cross_entropy(p, q):
    """Calculate cross_entropy H(p, q). refer to wiki https://en.wikipedia.org/wiki/Cross_entropy

    Args:
      p: tensor with dimension ~x1
      q: tensor with dimension ~x1
    """
    H = -p*tf.log(q) -(1-p)*tf.log(1-q)
    return tf.reduce_mean(H)
 
def img_listup(*img_list):

  if len(img_list) == 0:
    return None

  img_list_ = [a.T.astype(np.uint8) for a in img_list]

  h = sum([ a.shape[0] + 1 for a in img_list_])
  w = img_list_[0].shape[1]

  img = np.zeros((h, w), dtype=np.uint8)
  offset = 0
  for a in img_list_:
    end  = offset + a.shape[0]
    img[offset:end, :] = a*255
    offset = end + 1

  img_ = cv2.applyColorMap(img, cv2.COLORMAP_JET)

  img_ = cv2.resize(img_, (10*w, 10*h), interpolation=cv2.INTER_NEAREST)
  return img_

def main(args):

  seq_max =  FLAGS.in_seq_max_len*2 + 1 # 1 for delimiter
  ctrl = Controller(x_size=FLAGS.x_size, h_size=100, w_mem=FLAGS.W, num_layers=FLAGS.num_layers)
  ntm = NTMCell(y_size=FLAGS.y_size, batch_size=FLAGS.batch_size, num_step=(seq_max), Controller=ctrl, n_mem=FLAGS.N, R=1, W=1, dtype=tf.float32)

  x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, seq_max, FLAGS.x_size])
  y = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, seq_max, FLAGS.y_size])
  out = ntm(x)

  y_ = tf.reshape(y, shape=[-1, 1])
  out_ = tf.reshape(out, shape=[-1, 1])

  loss = cross_entropy(out_, y_)
  
  opt = get_opt(loss, "/*")
 
  print "====================================================="
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  
  start = datetime.now()
  print "Start: ",  start.strftime("%Y-%m-%d_%H-%M-%S")

  with tf.Session() as sess:
    # Initialize the variables (the trained variables and the
    sess.run(init_op)

    # Start input enqueue threads.
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(FLAGS.save_dir)
    print "checkpoint: %s" % checkpoint
    if checkpoint:
      print "Restoring from checkpoint", checkpoint
      saver.restore(sess, checkpoint)
    else:
      print "Couldn't find checkpoint to restore from. Starting over."
      dt = datetime.now()
      filename = "checkpoint" + dt.strftime("%Y-%m-%d_%H-%M-%S")
      checkpoint = os.path.join(FLAGS.save_dir, filename)

    for epoch in xrange(FLAGS.max_epoch):
      print "#####################################################################"
      for itr in xrange(FLAGS.max_itrs):
        print "==================================================================="
        print "[", epoch, "]", "%d/%d"%(itr, FLAGS.max_itrs)

        x_data, y_data = generate_data(FLAGS.x_size, FLAGS.y_size, seq_max, FLAGS.batch_size)
        feed_dict = {x:x_data, y:y_data}
        _, loss_val = sess.run([opt, loss], feed_dict)
        
        print "loss:", loss_val
        
        if itr > 1 and itr % 10 == 0:
          out_val = sess.run(out[0], feed_dict)

          result = img_listup(x_data[0], y_data[0], out_val)
          cv2.imshow('out', result)
          import scipy.misc
          #scipy.misc.imsave("generated"+current.strftime("%Y%m%d_%H%M%S")+".png", contrastive_sample_val)
          cv2.imwrite(FLAGS.save_dir + "/generated"+"%02d"%((itr/10)%100)+".png", result)
        cv2.waitKey(5)

        current = datetime.now()
        print "\telapsed:", current - start

        if itr > 1 and itr % FLAGS.save_itr == 0:
          print "#######################################################"
          saver.save(sess, checkpoint)

if __name__ == "__main__":
  tf.app.run()

