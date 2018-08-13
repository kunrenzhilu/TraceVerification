import argparse
import os

def get_data_parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--CITY', type=str, default='TKY', choices=['NYC', 'TKY'])
    parser.add_argument('--save_dir', type=str, default='data')
    parser.add_argument('--partition_rate', type=str, default='8:2')
    parser.add_argument('--partition_mode', type=str, default='random', choices=['random','byuser'])
    parser.add_argument('--ROOT', type=str, default='/home/haibin/oyk/checkins', help='root dir')   
    args = parser.parse_args(args)
    return args

def get_parser(args):
    parser = argparse.ArgumentParser()
    # pipeline
    parser.add_argument('--K', type=int, default=1, help='positive sample:negative sample=1:K')
    parser.add_argument('--num_worker', type=int, default=3)
    parser.add_argument('--CAPACITY', type=int, default=10)
    
    # directory
    parser.add_argument('--ROOT', type=str, default='/home/haibin/oyk/checkins', help='root dir')    
    parser.add_argument('--LOG_DIR', type=str, default='/home/haibin/oyk/checkins/SkipGram/log', help='experiment log directory')
    parser.add_argument('--CITY', type=str, default='TKY', choices=['NYC', 'TKY'], help='which city')
    parser.add_argument('--WITH_TIME', default=False, action='store_true',  help='whether the time is included')
    parser.add_argument('--WITH_USERID', default=False, action='store_true', help='whether to include the useid or not')
    parser.add_argument('--WITH_GPS', default=False, action='store_true', help='whether to include the gps or not')
    
    # model structure
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--vocabulary_size', type=int, default=-1, help='please retrive the number of node accordingly')
    parser.add_argument('--emb_dim', type=int, default=100, help='embedding dimension, default 100')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dim of rnn cells, default 256')
    parser.add_argument('--n_layers', type=int, default=2, help='number of hidden layer of rnn, default 2')
    parser.add_argument('--use_residual', default=False, action='store_true')
    parser.add_argument('--train_emb', default=False, action='store_true')
    
    # training
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training')
    parser.add_argument('--num_steps', type=int, default=int(5e4+1), help='the number of training steps, default 3e6')
    parser.add_argument('--num_epoch', type=int, default=10, help='the number of training epoch, should be used exclusively with the num_steps, default 50')
    parser.add_argument('--patten', type=str, default='day-24', choices=['day-6', 'day-24', 'week-end', 'weekly'], help='Used when --WITH_TIME, default day-24')
    parser.add_argument('--normalize_weight', default=False, action='store_true', help='whether to normalize the embeddings during training')
    parser.add_argument('--device', default='', type=str, help='which gpu device to use')
    #-- Evaluation
    parser.add_argument('--eval_batch_size', type=int, default=256, help='eval batch_size, default 256')
    args = parser.parse_args(args)
    
    if args.patten == 'day-6':
        args.n_timeslot = 6
    elif args.patten == 'day-24':
        args.n_timeslot = 24
    elif args.patten == 'weekly':
        args.n_timeslot = 7
    elif args.patten == 'week-end':
        args.n_timeslot = 2
    else: 
        args.n_timeslot = -1
    args.TB_DIR = os.path.join(args.LOG_DIR, 'tb')
    #args.EMB_DIR = os.path.join(args.LOG_DIR
    return args
    
    
