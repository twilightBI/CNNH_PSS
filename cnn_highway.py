# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:59:20 2017

@author: Dell
"""
import tensorflow as tf
from layers import multi_channel_cnn, linear

class cnn_highway(object):  
    def __init__(self, config, is_training = True):
        num_steps = config.num_steps
        keep_prob = config.keep_prob
        word_dim = config.word_dim
        batch_size = config.batch_size
        self.is_training = is_training
        
        #construct data
        self._X = tf.placeholder(tf.float32, shape=[None, num_steps, word_dim], name="seq")
        self._y = tf.placeholder(tf.float32, shape=[None, num_steps, config.class_dim], name="labels")
        
        #construct cnn model
        #the number of filter_hs is better to be an odd number
        def cnn(cnn_input, channel_num, num_layers, cnn_batch_size = batch_size):
            filter_hs = config.filter_hs
            layer_input = cnn_input
            img_h = layer_input.get_shape()[1].value
            with tf.variable_scope("cnn_layer{0}".format(0)):
                img_w = layer_input.get_shape()[2].value
                cnn_filter_input = tf.reshape(layer_input, [-1, img_h, img_w, 1]) 
                layer_output = multi_channel_cnn(
                    cnn_filter_input, img_h, img_w, filter_hs, cnn_batch_size, channel_num[0])
                layer_input = layer_output
                if self.is_training and config.keep_prob < 1:
                    layer_input = tf.nn.dropout(layer_input, keep_prob=keep_prob)
            for i in range(1, num_layers):
                input_dim = layer_input.get_shape()[2].value
                layer_input_b = tf.reshape(layer_input, [-1, input_dim])
                with tf.variable_scope("cnn_gates"):  
                    if i > 1:
                        tf.get_variable_scope().reuse_variables()
                    u = linear([layer_input_b], input_dim, True, 1.0, scope="gate")
                    u = tf.sigmoid(u)                 
                input_shape = layer_input.get_shape().as_list()
                
                with tf.variable_scope("cnn_layer{0}".format(i)):
                    img_w = layer_input.get_shape()[2].value
                    cnn_layer_input = tf.reshape(layer_input, [-1, img_h, img_w, 1]) 
                    layer_output = multi_channel_cnn(
                        cnn_layer_input, img_h, img_w, filter_hs, cnn_batch_size, channel_num[i])
                    if self.is_training and config.keep_prob < 1:
                        layer_output = tf.nn.dropout(layer_output, keep_prob=keep_prob)
                    layer_output_b = tf.reshape(layer_output, [-1, input_dim])   
                    layer_output = u * layer_input_b + (1-u)*layer_output_b
                    layer_output = tf.reshape(layer_output, input_shape)
                    layer_input = layer_output
            cnn_out = layer_input
            return cnn_out        
 
        cnn_out = cnn(self._X, config.cnn_channel_num, config.cnn_num_filters)
        fc_word_dim = len(config.filter_hs)*config.cnn_channel_num[-1]+word_dim
        fc_input = tf.reshape(tf.concat(axis=2, values=[cnn_out, self._X]), [-1, fc_word_dim])
        fc_out = linear([fc_input], config.fc_hidden_dim, True, 1.0, scope="fc")
        sx_input = tf.nn.relu(fc_out)
        scores = linear([sx_input], config.class_dim, True, 1.0, scope="softmax")
        
        self._pred = tf.arg_max(scores, dimension=1)

        if not self.is_training:
            return
            
        y_ = tf.reshape(self._y, [-1, config.class_dim]) 
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores,labels= y_)) 
        self._acc = self.count_acc(scores, y_, config)
        
        for var in tf.trainable_variables():
            reg_loss = config.reg * tf.nn.l2_loss(var)   
            tf.add_to_collection("total_loss", reg_loss)
        tf.add_to_collection("total_loss", self._cost)
        total_loss = tf.add_n(tf.get_collection("total_loss"))
        
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(list(zip(grads, tvars)))
        
        self._new_lr = tf.placeholder(tf.float32, shape=[], name= "new_lr")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        
    def count_acc(self, scores, y_, config):
        mask = tf.sign(tf.reduce_sum(y_[:, 0:8], reduction_indices=1))
        pred = tf.arg_max(scores, dimension=1)
        y = tf.arg_max(y_, dimension=1)
        correct = tf.cast(tf.equal(pred, y), tf.float32)
        correct *= mask
        acc = tf.reduce_sum(correct) / tf.reduce_sum(mask)
        return acc

    def add_feed_dict(self, X, y=None):
        feed_dict = {}
        feed_dict[self._X] = X
        if y is not None:
            feed_dict[self._y] = y
        return feed_dict
    
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr:lr_value})
    
    @property
    def initial_state(self):
        return (self._initial_state_fw, self._initial_state_bw)
    
    @property
    def cost(self):
        return self._cost

    @property
    def acc(self):
        return self._acc
    
    @property
    def predict(self):
        return self._pred
    
    @property
    def final_state(self):
        return self._final_state
    
    @property
    def lr(self):
        return self._lr
    
    @property
    def train_op(self):
        return self._train_op
    
