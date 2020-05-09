# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:19:14 2016

@author: hlt-titan
"""
import numpy as np
import tensorflow as tf
from cnn_highway import cnn_highway
from data_reader import get_sample_data, load_data
import time
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

class Config(object):
    class_dim = 9
    embed_size = 22
    pssm_size = 22
    word_dim = 22+22+2
    
    init_scale=1e-1
    learning_rate = 2e-3
    num_steps = 700
    fc_hidden_dim = 100
    filter_hs = [7, 9, 11]
    cnn_num_filters = 3
    cnn_channel_num = [100]*cnn_num_filters

    keep_prob = 0.5
    lr_decay = 0.95
    decay_epoch = 3
    max_grad_norm = 3

    batch_size = 10
    num_epoch = 500
    epoch_steps = 200
    print_ = 50
    collect_ = 10
    reg = 2e-5



def run_epoch(session, model, config, data, epoch, verbose=False):
    start_time = time.time()
    fetches = {}
    fetches["cost"] = model.cost
    fetches["acc"] = model.acc
    fetches["train_op"] = model.train_op
    step_basis = epoch*config.epoch_steps
    
    loss_vec = []
    for step in range(config.epoch_steps):
        if step>0:
            tf.get_variable_scope().reuse_variables()
        sample_X, sample_y = get_sample_data(data, config.batch_size)
        feed_dict = model.add_feed_dict(sample_X, sample_y)
        vals = session.run(fetches, feed_dict=feed_dict)
        loss = vals["cost"]
        acc = vals["acc"]
        if verbose and step % config.print_ == 0:
            print(("step: %d acc: %.3f loss: %.3f speed: %0f s time: %.3f s" %
                   (step_basis+step, acc, loss, (time.time()-start_time)/(config.print_*config.batch_size),
                    time.time()-start_time)))
            start_time = time.time()
        if step % config.collect_ == 0:
            loss_vec.append(loss)
    return loss_vec, acc


def validate(sess, model, config, val_data, best_val_acc, lr):
    val_batch = config.batch_size
    X = val_data["X"]
    y = val_data["y"]
    iters = X.shape[0] // val_batch
    iters = iters+1 if X.shape[0]%val_batch!=0 else iters
    num_vec = np.zeros(config.class_dim)
    correct_vec = np.zeros(config.class_dim)
    final_pred = []
    for it in range(iters):
        sample_X = np.zeros((val_batch, X.shape[1], X.shape[2]))
        sample_y = np.zeros((val_batch, y.shape[1], y.shape[2]))
        sample_y[:,:,8]=1
        idx_start = it*val_batch
        idx_end = idx_start+val_batch
        size = val_batch if idx_end <= X.shape[0] else X.shape[0]-idx_start
        sample_X[:size] = X[idx_start:idx_end]
        sample_y[:size] = y[idx_start:idx_end]
        feed_dict = model.add_feed_dict(sample_X)
        sample_pred = sess.run(model.predict, feed_dict=feed_dict)
        sample_y = np.argmax(sample_y, axis=2)
        sample_y = sample_y.reshape(-1)
        final_pred.append(sample_pred)
        for i in range(config.class_dim-1):
            idxes = np.where(sample_y == i)[0]
            num_vec[i] += idxes.shape[0]
            correct_vec[i] += np.sum(sample_pred[idxes] == i)
            
    total_acc = np.sum(correct_vec) / np.sum(num_vec)
    L_acc = correct_vec[0] / float(num_vec[0])
    B_acc = correct_vec[1] / float(num_vec[1])
    E_acc = correct_vec[2] / float(num_vec[2])
    G_acc = correct_vec[3] / float(num_vec[3])
    I_acc = correct_vec[4] / float(num_vec[4])
    H_acc = correct_vec[5] / float(num_vec[5])
    S_acc = correct_vec[6] / float(num_vec[6])
    T_acc = correct_vec[7] / float(num_vec[7])
    print("---------------------------------------------------------------------------------------")
    print("predict: ")
    print("val acc: %.3f best val acc: %.3f lr: %.3e" %(total_acc, best_val_acc, lr))
    print("L acc: ", L_acc, ", B acc: ", B_acc, "E acc: ", E_acc, ", G acc: ", G_acc)
    print("I acc: ", I_acc, ", H acc: ", H_acc, "S acc: ", S_acc, ", T acc: ", T_acc)
    print("---------------------------------------------------------------------------------------")
    print("\n")
    return total_acc

if __name__ == "__main__":
    debug = False
    config = Config()
    train_data, val_data, test_data = load_data(debug, config.num_steps)
    print("-----------------------------------------------------")
    print("train data shapes")
    print("train X: ", train_data["X"].shape)    
    print("train label: ", train_data["y"].shape)
    print("val data shapes")
    print("val X: ", val_data["X"].shape)    
    print("val label: ", val_data["y"].shape)
    print("test data shapes")
    print("test X: ", test_data["X"].shape)    
    print("test label: ", test_data["y"].shape)
    print("-----------------------------------------------------")
    
    lr = config.learning_rate
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    best_val_acc=0; not_improve = 0
    train_loss_his = []; train_acc_his = []; val_acc_his = []
    with tf.Graph().as_default():
        with tf.name_scope("Train"):
            with tf.variable_scope("model", reuse=None,initializer=initializer):
                model = cnn_highway(config, is_training=True)
        with tf.name_scope("Val"):
            with tf.variable_scope("model", reuse=True,initializer=initializer):
                val_model = cnn_highway(config, is_training=False)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())
            for epoch in range(config.num_epoch):
                if epoch>0 and lr>1e-5 and epoch%config.decay_epoch==0:
                    lr *= config.lr_decay
                model.assign_lr(sess, lr)
                loss_vec, train_acc = run_epoch(sess, model, config, train_data, epoch, verbose=True)
                val_acc = validate(sess, val_model, config, val_data, best_val_acc, lr)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    saver.save(sess, "./model/best_model.weights")
                    not_improve=0
                else:
                    not_improve += 1
                if not_improve >= 5 and lr>1e-5:
                    lr /= 2
                    not_improve=1
    
