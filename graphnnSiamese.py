import tensorflow as tf
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score

def graph_embed(X, msg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
    """
    : param msg_mask:    (?, ?, ?) (neighborhood mask)
    : param N_x:         7         (# of features)
    : param N_embed:     64        (embed dimension)
    : param N_o:         64        (output dimension)
    : param iter_level:  5         (iteration level)
    : param Wnode:       (7, 64)
    : param Wembed:      (64, 64), len=2
    : param W_output:    (64, 2)
    : param b_output:    (2, )
    : return output:     (?, 2)
    """
    # (?, ?, 64)
    node_val = tf.reshape(tf.matmul(tf.reshape(X, [-1, N_x]), Wnode), [tf.shape(X)[0], -1, N_embed])

    cur_msg = tf.nn.relu(node_val)
    
    for t in range(iter_level):
        # linear transformation
        # (?, ?, 64)
        Li_t = tf.matmul(msg_mask, cur_msg)
        # (?, ?, 64)
        cur_info = tf.reshape(Li_t, [-1, N_embed])

        for Wi in Wembed:
            if Wi == Wembed[-1]:
                cur_info = tf.matmul(cur_info, Wi)
            else:
                cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
        # reshape back to batch size
        neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
        # summing
        tot_val_t = node_val + neigh_val_t
        # non-linearity 
        tot_msg_t = tf.nn.tanh(tot_val_t)
        cur_msg = tot_msg_t
    
    g_embed = tf.reduce_sum(cur_msg, 1)
    output = tf.matmul(g_embed, W_output) + b_output
    
    return output


class graphnn(object):
    def __init__(self,
                    N_x,
                    Dtype, 
                    N_embed,
                    depth_embed,
                    N_o,
                    ITER_LEVEL,
                    lr,
                    device = '/gpu:0'
                ):

        self.NODE_LABEL_DIM = N_x

        tf.compat.v1.reset_default_graph()
        with tf.device(device):
            Wnode = tf.Variable(tf.random.truncated_normal(
                shape = [N_x, N_embed], stddev = 0.1, dtype = Dtype))
            Wembed = []
            for i in range(depth_embed):
                Wembed.append(tf.Variable(tf.random.truncated_normal(
                    shape = [N_embed, N_embed], stddev = 0.1, dtype = Dtype)))

            W_output = tf.Variable(tf.random.truncated_normal(
                shape = [N_embed, N_o], stddev = 0.1, dtype = Dtype))
            b_output = tf.Variable(tf.constant(0, shape = [N_o], dtype = Dtype))
            
            X1 = tf.compat.v1.placeholder(Dtype, [None, None, N_x]) #[B, N_node, N_x]
            msg1_mask = tf.compat.v1.placeholder(Dtype, [None, None, None]) #[B, N_node, N_node]
            
            self.X1 = X1
            self.msg1_mask = msg1_mask
            embed1 = graph_embed(X1, msg1_mask, N_x, N_embed, N_o, ITER_LEVEL,
                    Wnode, Wembed, W_output, b_output)  #[B, N_x]

            X2 = tf.compat.v1.placeholder(Dtype, [None, None, N_x])
            msg2_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
            self.X2 = X2
            self.msg2_mask = msg2_mask
            embed2 = graph_embed(X2, msg2_mask, N_x, N_embed, N_o, ITER_LEVEL,
                    Wnode, Wembed, W_output, b_output)

            label = tf.compat.v1.placeholder(Dtype, [None, ]) #same: 1; different:-1
            self.label = label
            self.embed1 = embed1

            # cosine distance
            cos = tf.reduce_sum(embed1 * embed2, 1) / tf.sqrt(tf.reduce_sum(
                embed1 ** 2, 1) * tf.reduce_sum(embed2 ** 2, 1) + 1e-10)

            diff = -cos
            self.diff = diff
            # loss function
            loss = tf.reduce_mean( (diff + label) ** 2 )
            self.loss = loss

            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            self.optimizer = optimizer
    
    def say(self, string):
        print(string)
        if self.log_file != None:
            self.log_file.write(string+'\n')
    
    def init(self, LOAD_PATH, LOG_PATH):
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = False
        sess = tf.compat.v1.Session(config=config)
        saver = tf.compat.v1.train.Saver(max_to_keep=101)
        self.sess = sess
        self.saver = saver
        self.log_file = None
        if (LOAD_PATH is not None):
            if LOAD_PATH == '#LATEST#':
                checkpoint_path = tf.train.latest_checkpoint('./')
            else:
                checkpoint_path = LOAD_PATH
            saver.restore(sess, checkpoint_path)
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'a+')
            self.say('{}, model loaded from file: {}'.format(
                datetime.datetime.now(), checkpoint_path))
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'w')
            self.say('Training start @ {}'.format(datetime.datetime.now()))
    
    def get_embed(self, X1, mask1):
        vec, = self.sess.run(fetches=[self.embed1],
                feed_dict={self.X1:X1, self.msg1_mask:mask1})
        return vec

    def calc_loss(self, X1, X2, mask1, mask2, y):
        cur_loss, = self.sess.run(fetches=[self.loss], feed_dict={self.X1:X1,
            self.X2:X2,self.msg1_mask:mask1,self.msg2_mask:mask2,self.label:y})
        return cur_loss
        
    def calc_diff(self, X1, X2, mask1, mask2):
        diff, = self.sess.run(fetches=[self.diff], feed_dict={self.X1:X1,
            self.X2:X2, self.msg1_mask:mask1, self.msg2_mask:mask2})
        return diff
    
    def train(self, X1, X2, mask1, mask2, y):
        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict={self.X1:X1,
            self.X2:X2,self.msg1_mask:mask1,self.msg2_mask:mask2,self.label:y})
        return loss
    
    def save(self, path, epoch=None):
        checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
        return checkpoint_path
