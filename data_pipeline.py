import numpy as np
from multiprocessing import Process, Queue 
from threading import Event
import random
import time
import gc
import tensorflow as tf
from train_utils import process_seq, TriadList
random.seed(3000)
pad_sequence = tf.keras.preprocessing.sequence.pad_sequences
        
def unreplace_sample(array, i):
    n = np.random.choice(array,)
    while i == n:
        n = np.random.choice(array)
    return n

class Sampler(Process):
    def __init__(self, trajs_dict, data_queue, event, id, K=1):
        super(Sampler, self).__init__()
        self.trajs_dict = trajs_dict
        self.K = K
        self.data_queue = data_queue
        self.stop_event = event
        self.epoch = 0
        self.id = id
        self.stack = list(trajs_dict.keys())
        random.shuffle(self.stack)
        
    def refill_queue(self):
        # gc.collect()
        self.stack = list(self.trajs_dict.keys())
        random.shuffle(self.stack)
        self.epoch += 1
        print('worker[{}] Epoch {}'.format(self.id, self.epoch))
        
    def sample(self,):
        if len(self.stack) <= 0:
            self.refill_queue()
        usr = self.stack.pop()
        trajs = self.trajs_dict[usr]
        traj_ids = list(range(len(trajs)))
        sample_list_left = []
        sample_list_right = []
        label_list = []
        
        for i in traj_ids:
            j = unreplace_sample(traj_ids, i)
            sample_list_left.append(process_seq(trajs[i]))
            sample_list_right.append(process_seq(trajs[j]))
            label_list.append([1])
            
            for k in range(self.K):
                usr_k = unreplace_sample(list(self.trajs_dict.keys()), usr)
                sample_list_left.append(process_seq(trajs[i]))
                trajs_k = self.trajs_dict[usr_k]
                id = np.random.choice(len(trajs_k))
                sample_list_right.append(process_seq(trajs_k[id]))
                label_list.append([0])
        assert len(sample_list_left) == len(sample_list_right), 'left:right={}:{}'.format(len(sample_list_left), len(sample_list_right))
        assert len(sample_list_left) == len(label_list), 'left:label={}:{}'.format(len(sample_list_left), len(label_list))
        return sample_list_left, sample_list_right, label_list
    
    def run(self, debug=False):
        if debug:
            debugstr = 'Debug [{}]'.format(self.id)
            print('{} Stop event set'.format(debugstr, self.stop_event.is_set()))
            print('{} Data queue is full'.format(debugstr, self.data_queue.full()))
            print('{} Try putting to data queue'.format(debugstr))
            self.data_queue.put(self.sample())
            print('{} Putted, check data queue is full: {}'.format(debugstr, self.data_queue.full()))
        while not self.stop_event.is_set():
            if not self.data_queue.full():
                #print('{} puts [{}/{}]'.format(self.id, len(self.stack), len(self.trajs_dict)))
                self.data_queue.put(self.sample())
            else:
                time.sleep(0)
#                print('discre queue is full')
        
class BatchAggregator(Process):
    def __init__(self, input_queue, output_queue, stop_event, args):
        super(BatchAggregator, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.pipe = TriadList()
        self.batch_size = args.batch_size
        self.pad_value = args.pad_value
        
    def collate_fn(self, x, pad_value):
        left, right, labels = x
        left = pad_sequence(left, padding='post', value=pad_value)
        right = pad_sequence(right, padding='post', value=pad_value)
        return (left, right, labels)
    
    def run(self):
        while not self.stop_event.is_set():
            while not len(self.pipe) >= self.batch_size:
                self.pipe.extend(self.input_queue.get())
#                print('get')
            if not self.output_queue.full():
                tmp_triad = self.pipe.pop(self.batch_size)
                if len(tmp_triad[0]) == 0: 
                    print('Aggregator got zero len triple, pipe len {}, continue'.format(len(self.pipe)))
                    continue
                self.output_queue.put(self.collate_fn(tmp_triad, self.pad_value))
                
    def terminate(self):
        import gc
        self.pipe = None
        gc.collect()
        super(BatchAggregator, self).terminate()

class SampleManager:
    def __init__(self, data, args):
        self.stop_event = Event()
        self.discre_queue = Queue(args.CAPACITY*3)
        self.data_queue = Queue(args.CAPACITY)
        self.samplers = {i:Sampler(data, self.discre_queue, self.stop_event, i, args.K) for i in range(args.num_worker)}
        self.batchagr = BatchAggregator(self.discre_queue, self.data_queue, self.stop_event, args)
        self.max_batch = self.compute_max_batch_num(data, args)
        self.n_step = 0
        self.n_epoch = 0
    
    def compute_max_batch_num(self, data, args):
        tlt_len = sum([len(v) for v in data.values()])
        max_batch = int(1 + (tlt_len*(1+args.K))/args.batch_size)
        return max_batch
    
    def start(self,):
        print('Starting {} samplers..'.format(len(self.samplers)))
        for s in self.samplers.values(): 
            s.start()
        print('samplers {} has been started'.format([s.pid for s in self.samplers.values()]))    
        
        self.batchagr.start()
        print('batch aggregator {} has been started'.format(self.batchagr.pid))
        
    def stop(self, timeout=5):
        self.stop_event.set()
        for s in self.samplers.values():
            s.join(timeout)
            s.terminate()
        self.batchagr.join(timeout)
        self.batchagr.terminate()
        terminate_state = self.assure_terminated()
        print('All threads have been terminated: ', terminate_state)
        return terminate_state

    def assure_terminated(self):
        for i, s in self.samplers.items():
            if s.is_alive():
                print('sampler {} is still alive'.format((i, s.pid)))
                return False
        if self.batchagr.is_alive():
            print('aggregator {} is still alive'.format(self.batchagr.pid))
            return False
        return True
    
    def next_batch(self, timeout=5):
        self.n_step += 1
        try:
            return self.data_queue.get(timeout)
        except:
            for k, v in self.samplers.items():
                print('{} is alive: {}'.format(k, v.is_alive()))
                v.run(debug=True)
            print('discre_queue is full: {} , empty: {}, queue_size: {}'.format(self.discre_queue.full(), self.discre_queue.empty(), self.discre_queue.qsize()))
            print('data_queue is full: {} , empty: {}, queue_size: {}'.format(self.data_queue.full(), self.data_queue.empty(), self.data_queue.qsize()))
            self.stop()
            raise
    
    def get_current_epoch(self,):
        return int(self.n_step/self.max_batch)
    
    def get_current_batch(self,):
        return self.n_step