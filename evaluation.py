from train_utils import process_seq, TriadList
from data_pipeline import pad_sequence
import sklearn
import numpy as np
from collections import defaultdict
import time
from sklearn import metrics
import os
import pickle

prfs_metric = metrics.precision_recall_fscore_support
accuracy_metric = metrics.accuracy_score

def process_testset(test_data, test_mode):
    def postprocess_data(tmp_list, label):
        res_list = []
        for seq_tuple in tmp_list:
            res_list.append((process_seq(seq_tuple[0]), process_seq(seq_tuple[1]), [label]))
        return res_list

    def convert_tuples_to_batchs(tuple_list):
        tri_list = TriadList()
        for tupl in tuple_list:
            tri_list.append(tupl)
        return tri_list
    genuine_set = test_data[True]
    imposter_set= test_data[False]
    
    res_list = []
    res_list += postprocess_data(test_data[True], 1)
    if test_mode == 'small':
        res_list += postprocess_data(test_data[False][:len(genuine_set)], 0)
    elif test_mode == 'middle':
        res_list += postprocess_data(test_data[False][:len(genuine_set)*10], 0)
    elif test_model == 'full':
        res_list += postprocess_data(test_data[False], 0)
    else:
        raise ValueError('test_mode not in [small, middle, full]')
    tri_list = convert_tuples_to_batchs(res_list)
    return tri_list

class TestDataGenerator:
    def __init__(self, test_data, args):
        self.test_data = process_testset(test_data, 'small')
        self.max_len = len(self.test_data)
        self.args = args
        self.pad_value = args.pad_value
        self.reset()
        
    def next_batch(self, batch_size):
        l, r, lbl = [], [], []
        for idx in self.indices[self.i:self.i+batch_size]:
            a, b, c = self.test_data[idx]
            l.append(a)
            r.append(b)
            lbl.append(c)
        '''if (len(l) == 0) or (len(r) == 0):
            print('len(l): {}, len(r): {}'.format(len(l), len(r)))
            print('i: {}, batch_size: {}, max_len {}'.format(self.i, batch_size, self.max_len))
            print(self.indices[self.i:self.i+batch_size])
            a, b, c = self.test_data[0]
            l.append(a)
            r.append(b)'''
        l = pad_sequence(l, padding='post', value=self.pad_value)
        r = pad_sequence(r, padding='post', value=self.pad_value)
        self.i += batch_size
        done = (self.i >= self.max_len)
        return (done, (l, r, lbl))
    
    def reset(self):
        self.i = 0
        self.indices = list(range(len(self.test_data)))
        #np.random.shuffle(self.indices) # no need to shuffle in test.

class Evaluator(object):
    def __init__(self, dg, args):
        self.dg = dg
        self.log_dir = args.LOG_DIR
        self.history = defaultdict(list)
        self.args = args
        
    def evaluate(self, model, sess):
        yhat_list = []
        label_list = []
        res_dict = defaultdict(list)
        done = False
        tick = time.time()
        
        while not done:
            done, (left, right, label) = self.dg.next_batch(self.args.batch_size)
            batch_size = len(left)
            predicts = sess.run(model.predicts, feed_dict={model.bs_phd:batch_size, model.left_phd:left, model.right_phd:right})
            yhat_list.extend(predicts)
            label_list.extend(label)
            
        self.dg.reset() #important
        yhat_list = np.concatenate(yhat_list)
        label_list = np.concatenate(label_list)
        
        p, r, f, s = prfs_metric(y_true=label_list, y_pred=yhat_list)
        res_dict['precision'], res_dict['recall'], res_dict['f1'], res_dict['support']\
            = p[1], r[1], f[1], s[1] # this is because p[0] relates to the pos_label=0, p[1] relates to teh pos_label=1
        res_dict['accuracy'] = accuracy_metric(y_true=label_list, y_pred=yhat_list)
        res_dict['t'] = time.time()-tick
        return res_dict
    
    def update_history(self, loss=None, res_dict=None):
        if not loss is None:
            self.history['loss'].append(loss)
        if not res_dict is None:
            for k, v in res_dict.items():
                self.history[k].append(v)
        
    def save_history(self):
        save_path = os.path.join(self.log_dir, 'history.pk')
        with open(save_path, 'wb') as f:
            pickle.dump(self.history, f)
        print('saved history to {}'.format(save_path))
    
    def load_history(self, args):
        path = os.path.join(args.LOG_DIR, 'history.pk')
        print('load history from {}'.format(path))
        with open(path, 'rb') as f:
            self.history = pickle.load(f)
            
    @staticmethod
    def load(path):
        print('load history from {}'.format(path))
        with open(path, 'rb') as f:
            return pickle.load(f)