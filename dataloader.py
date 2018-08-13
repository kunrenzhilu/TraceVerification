import os
import sys
version = sys.version_info.major
if version == 3:
    import pickle
else:
    import cPickle as pickle
import numpy as np
from collections import defaultdict
from errata import correct_errata

ROOT = '/home/haibin/oyk/checkins/data'
CITY = 'NYC'
class Dictionary(object):
    def __init__(self,):
        pass

def load_category_mapping(path=None):
    if path is None:
        path = os.path.join('data', 'category_mappings.pk')
    with open(path, 'rb') as f:
        both_dict = pickle.load(f)
    return both_dict

def get_rootctgy_chkin(ctgy_chkin_dict, subctgy_ctgy_dict):
    newdict = defaultdict(list)
    for key in ctgy_chkin_dict:
        newdict[subctgy_ctgy_dict[correct_errata(key)]].extend(ctgy_chkin_dict[key])
    return newdict

def load_local_dataset(path=None):
    assert os.path.isfile(path), '{} is not a file'.format(path)
    import time
    tick = time.time()
    with open(path, 'rb') as f:
        train, test = pickle.load(f)
    print('Load data from {}, used time: {}'.format(path, time.time()-tick))
    return train, test    

def load_data(data_path=None):
    assert os.path.isfile(data_path), '{} is not a file'.format(data_path)
    
    dicts = Dictionary()
    with open(data_path, 'rb') as f:
        data_from_dump = pickle.load(f)
    data = data_from_dump['data']
    dicts.dictionary = data_from_dump['dictionary']
    dicts.reverse_dictionary = data_from_dump['reverse_dictionary']
    dicts.ctgy_len_arrays = data_from_dump['ctgy_len_arrays']
    dicts.vocabulary_size = data_from_dump['vocabulary_size']
    dicts.chkin_ctgy_dict = data_from_dump['chkin_ctgy_dict']
    dicts.ctgy_chkin_dict = data_from_dump['ctgy_chkin_dict']
    dicts.chkin_len_arrays = data_from_dump['chkin_len_arrays']
    dicts.ctgy_mapping = load_category_mapping(os.path.join(ROOT, 'category_mappings.pk'))
    dicts.rootctgy_chkin_dict = get_rootctgy_chkin(dicts.ctgy_chkin_dict, dicts.ctgy_mapping['subctgy_ctgy_dict'])
    
    #keep the dictionary's key set fixed
    for k, v in dicts.__dict__.items():
        if type(v) == defaultdict:
            v.default_factory = None
    return data, dicts


def keep_data_by_uid(data, uids):
    trajs = list()
    for seq in data:
        tmpseq = [] 
        if seq[0][1] in uids:
            for point in seq:
                tmpseq.append(point[0])
        if len(tmpseq) > 0:
            trajs.append(tmpseq)
    return trajs 