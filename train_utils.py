import os
import json

class TriadList:
    def __init__(self):
        self.left = list()
        self.right = list()
        self.label = list()
        
    def __len__(self,):
        return len(self.left)
    
    def extend(self, triad):
        l, r, lbl = triad
        self.left.extend(l)
        self.right.extend(r)
        self.label.extend(lbl)
        
    def append(self, triad):
        l, r, lbl = triad
        self.left.append(l)
        self.right.append(r)
        self.label.append(lbl)
        
    def is_empty(self,):
        return len(self.left) == 0
    
    def pop(self, size):
        l, r, lbl = [], [], []
#         size = min(size,  self.__len__())
        for i in range(size):
            l.append(self.left.pop(0))
            r.append(self.right.pop(0))
            lbl.append(self.label.pop(0))
        return (l, r, lbl)
    
    def __getitem__(self, i):
        l = self.left[i]
        r = self.right[i]
        lbl = self.label[i]
        return (l, r, lbl)
    
def save_model_tf(saver, sess, args, path=None):
    save_path = os.path.join(args.LOG_DIR, 'saved', 'model.ckpt') if path is None else path
    saver.save(sess, save_path)
    print('Saved model to {}'.format(save_path))
    
def load_model_tf(saver, args, sess, path=None):
    load_path = os.path.join(args.LOG_DIR, 'saved', 'model.ckpt') if path is None else path
    assert os.path.isfile(load_path+'.meta'), '{} is empty'.format(load_path)
    saver.restore(sess, load_path)
    return sess

def process_seq(seq):
    res_lt = []
    for point in seq:
        res_lt.append(point[0])
    return res_lt

def save_args(args):
    with open(os.path.join(args.LOG_DIR, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    print('Saved args to {}'.format(f.name))
def load_args(path):
    with open(path, 'r') as f:
        args = json.load(f)
    print(args)
    return args

def complete_args(args, dicts):
    args.vocabulary_size = dicts.vocabulary_size
    args.pad_value = args.vocabulary_size + 1
    return args