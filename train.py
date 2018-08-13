import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import copy
import time
import random
import sys
import argparse
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import random
random.seed(1000)
np.random.seed(1000)

from parser import get_parser
from dataloader import load_local_dataset, load_data
from data_pipeline import SampleManager, Sampler, BatchAggregator
from logger import Logger
from train_utils import save_model_tf, load_model_tf, save_args, load_args, complete_args
from model import Model, init_params
from evaluation import TestDataGenerator, Evaluator

def train(graph, sess, model, evaluator, logger, manager, args):
    # ----prepare
    save_args(args)
    n_epoch = 0
    n_batch = 0
    losses = []
    manager.start()
    
    # ----start training
    with graph.as_default():
        saver = tf.train.Saver(model.all_params)
        if args.resume:
            sess = load_model_tf(saver, args, sess)
        else: 
            sess.run(tf.global_variables_initializer())  
        logger.log('\nStart training')
        try:
            while n_epoch < args.num_epoch:
                epoch_tick = time.time()
                # evaluate and save model
                result = evaluator.evaluate(model, sess)
                evaluator.update_history(res_dict=result)
                evaluator.save_history()
                save_model_tf(saver, sess, args)
                
                while n_epoch >= manager.get_current_epoch():
                    left, right, labels = manager.next_batch()
                    tmp_loss, _ = sess.run([model.loss, model.optimize], 
                       feed_dict={model.bs_phd:args.batch_size, model.left_phd: left, model.right_phd: right, model.label_phd: labels})
                    losses.append(tmp_loss)
                    
                    #----- record and evaluate
                    if n_batch % 100 == 0:
                        loss = np.mean(losses)
                        logstr = '[{}] loss:{:.6f}'.format(n_batch, loss)
                        evaluator.update_history(loss=loss)
                        logger.log(logstr)
                        losses = []                       
                    n_batch += 1
                              
                n_epoch += 1
                logstr = '#'*50+'\n'
                logstr += 'Epoch {}, used time: {}, {}'.format(n_epoch, time.time()-epoch_tick, result)
                logger.log(logstr)
            manager.stop()
            
        except Exception as e:
            print(e)
            if not manager.stop():
                manager.stop()
            raise
    return sess

if __name__=='__main__':
    args = get_parser(sys.argv[1:])
    tick = time.time()
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    #load all necessary data
    local_data_path = 'data/{}_byuser.pk'.format(args.CITY) 
    train_set, test_set = load_local_dataset(local_data_path)
    data_path = os.path.join(
        os.path.join(args.ROOT, 'data','{}_INTV_processed_voc5_len2_setting_WITH_GPS_WITH_TIME_WITH_USERID.pk'.format(args.CITY)))
    _, dicts = load_data(data_path)
    args = complete_args(args, dicts)
    del _
    
    #preparation
    manager = SampleManager(train_set, args)
    test_dg = test_dg = TestDataGenerator(test_set, args)
    evaluator = Evaluator(test_dg, args)
    logger = Logger(os.path.join(args.LOG_DIR, 'log.txt'))

    graph = tf.Graph()
    with graph.as_default():
        model = Model(args)
        sess = tf.Session(graph=graph, config=config)
    train(graph, sess, model, evaluator, logger, manager, args)
    sess.close()
    manager.stop()
    print('Done, every thing in {}, time used {}'.format(args.LOG_DIR, time.time()-tick))