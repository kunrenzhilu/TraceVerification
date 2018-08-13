import tensorflow as tf
import numpy as np
from collections import defaultdict
import random


def init_params(shape, param):
    if param is None:
        return tf.random_uniform(shape=shape, minval=-1., maxval=1.)
    else: 
        return tf.constant(param)


class Model(object):
    def __init__(self, args, pretrained_emb=None):
        # ----------pretrain
        self.pretrained_emb = init_params((args.vocabulary_size, args.emb_dim), pretrained_emb)
        # ----------placeholder
        self.bs_phd = tf.placeholder(tf.int32, shape=(), name='batch_size_placeholder')
        self.left_phd = tf.placeholder(shape=(None, None), dtype=tf.int32, name='left_placeholder')
        self.right_phd = tf.placeholder(shape=(None, None), dtype=tf.int32, name='right_placeholder')
        self.label_phd = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        # ----------network
        self.embeddings = tf.get_variable('embeddings', initializer=self.pretrained_emb, trainable=args.train_emb)
        self.pad_emb = tf.get_variable('padding', shape=(1, args.emb_dim), trainable=True)
        self.all_embeddings = tf.concat([self.embeddings, self.pad_emb], axis=0)
        left_trajs = tf.nn.embedding_lookup(self.all_embeddings, self.left_phd)
        right_trajs = tf.nn.embedding_lookup(self.all_embeddings, self.right_phd)
        
        def encoder(inp, args, states=None, reuse=False):
            with tf.variable_scope('RNN_encoder', reuse=reuse):
                inp = tf.layers.dense(inp, args.hidden_dim, activation=tf.nn.leaky_relu)
                inp = tf.layers.dropout(inp)
            
                # build the rnn 
                cells = []
                for i in range(args.n_layers):
                    cell = tf.nn.rnn_cell.GRUCell(num_units=args.hidden_dim)  
                    if not i == args.n_layers - 1:
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
                    if args.use_residual:
                        cell = tf.nn.rnn_cell.ResidualWrapper(cell=cell)
                    cells.append(cell)

                stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
                init_states = stacked_cell.zero_state(self.bs_phd, tf.float32) if states is None else states
                out, hidden = tf.nn.dynamic_rnn(stacked_cell, inputs=inp, initial_state=init_states)
                return out, hidden
            
        def output_layer(left, right): #alternatively use softmax
            logits = tf.layers.dense(tf.abs(left-right), 1) #alternatively use l1/l2norm
            return logits
        
        # -- main computation starts
        left_out, left_hidden = encoder(left_trajs, args)
        right_out,right_hidden= encoder(right_trajs, args, reuse=True)
            
        self.left_latent = left_hidden[-1]
        self.right_latent = right_hidden[-1]
        out_logits = output_layer(self.left_latent, self.right_latent)
        
        self.output_prob = tf.sigmoid(out_logits)
        self.predicts = tf.cast(self.output_prob > 0.5, tf.int32)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_phd, logits=out_logits))
        # -- main computation ends
        
        self.trainable_params = tf.trainable_variables()
        self.all_params = tf.global_variables()
        
        global_step_g = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(args.lr, global_step_g, 1000, 0.1, staircase=True)    
        optimizer = tf.train.AdamOptimizer(learning_rate)
        
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimize = optimizer.apply_gradients(zip(gradients, variables))