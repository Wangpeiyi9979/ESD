import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import torch
import random

from util.data_loader import get_loader
from util.framework import FewShotNERFramework
from model.ESD import ESD

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='inter',
                        help='training mode, must be in [inter, intra, supervised, 1, 2, 3, 4, ...]')
    parser.add_argument('--dataset', default='fewnerd',
                        help='training datasets, must be in [fewnerd, snips, ner]')
    parser.add_argument('--trainN', default=5, type=int,
                        help='N in train')
    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=2, type=int,
                        help='K shot')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--model', default='ESD',
                        help='model name, must be basic-bert, proto, nnshot, or structshot')
    parser.add_argument('--max_length', default=100, type=int,
                        help='max length')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--bert_lr', default=3e-5, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='dropout rate')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')

    parser.add_argument('--max_epoch', default=10, type=int,
                        help='max_epoch')

    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
                        help='use nvidia apex fp16')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')


    parser.add_argument('--optimizer', type=str, default='adamw',
                    help='optimizer')

    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle train data')

    parser.add_argument('--hidsize', default=100, type=int,
                        help='dimension of hidden_size')

    parser.add_argument('--bert_path', default='bert-base-uncased', type=str,
                        help='bert-path')

    parser.add_argument('--max_o_num', default=10, type=int,
                        help='down-sampling for fewnerd for type O in each sentence')

    parser.add_argument('--num_heads', default=1, type=int,
                        help='multi-head-attention head num')

    parser.add_argument('--L', default=8, type=int,
                        help='multi-head-attention head num')

    parser.add_argument('--early_stop', default=6, type=int,
                        help='multi-head-attention head num')

    parser.add_argument('--val_step', default=1500, type=int)

    parser.add_argument('--soft_nms_k', default=1e-5, type=float)

    parser.add_argument('--soft_nms_u', default=1e-5, type=float)

    parser.add_argument('--soft_nms_delta', default=0.1, type=float)

    parser.add_argument('--beam_size', default=5, type=int)

    parser.add_argument('--warmup_step', default=1000, type=int)

    parser.add_argument('--eposide_tasks', default=1, type=int)

    opt = parser.parse_args()
    N = opt.N
    K = opt.K
    model_name = opt.model
    opt.O_class_num = 3

    if opt.dataset == 'fewnerd':
        print("{}-way-{}-shot Few-Shot NER".format(N, K))
    else:
        print("{}-shot SNIPS".format(K))
    print("model: {}".format(model_name))
    print('mode: {}'.format(opt.mode))

    set_seed(opt.seed)

    print('loading model and tokenizer...')

    print('loading data...')
    if opt.dataset == 'fewnerd':
        opt.train = f'data/{opt.mode}/train'
        opt.test = f'data/{opt.mode}/test'
        opt.val = f'data/{opt.mode}/dev'

    elif opt.dataset == 'snips':
        if K == 5:
            root_dataset = 'data/xval_' + opt.dataset +  '_shot_5'
            opt.train = f'{root_dataset}/{opt.dataset}-train-{opt.mode}-shot-5.json'
            opt.val = f'{root_dataset}/{opt.dataset}-valid-{opt.mode}-shot-5.json'
            opt.test = f'{root_dataset}/{opt.dataset}-test-{opt.mode}-shot-5.json'
        else:
            root_dataset =  'data/xval_' + opt.dataset
            opt.train = f'{root_dataset}/{opt.dataset}_train_{opt.mode}.json'
            opt.val = f'{root_dataset}/{opt.dataset}_valid_{opt.mode}.json'
            opt.test = f'{root_dataset}/{opt.dataset}_test_{opt.mode}.json'

    model = ESD(opt)
    prefix = opt.dataset + '-' + model.model_name
    if opt.dataset == 'fewnerd':
        prefix += f'-N_{opt.N}-K_{opt.K}-mode_{opt.mode}-drop_{opt.dropout}-lr_{opt.lr}-bertlr_{opt.bert_lr}-hidsize_{opt.hidsize}-graditer_{opt.grad_iter}-es_{opt.early_stop}-warmup_{opt.warmup_step}-eptasks_{opt.eposide_tasks}'
    else:
        prefix += f'-K_{opt.K}-mode_{opt.mode}-drop_{opt.dropout}-lr_{opt.lr}-bertlr_{opt.bert_lr}-hidsize_{opt.hidsize}-graditer_{opt.grad_iter}-es_{opt.early_stop}-warmup_{opt.warmup_step}-eptasks_{opt.eposide_tasks}'
    if opt.shuffle:
        prefix += '-sff'
    prefix += '-maxonum_{}'.format(opt.max_o_num)
    prefix += '-seed_{}'.format(opt.seed)
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)

    
    train_data_loader = get_loader(opt.train, opt, shuffle=opt.shuffle)

    opt.batch_size=1  # we force the batch_size = 1 during evaluation for fair comparison with baselines in SNIPS.
    val_data_loader = get_loader(opt.val, opt)
    test_data_loader = get_loader(opt.test, opt)

   
    framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, opt)

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')


    print("*" * 20)
    print(opt)
    print("*" * 20)
    print('*' * 20)
    print('[save_ckpt]: {}'.format(ckpt))
    print('*' * 20)
    if torch.cuda.is_available():
        model.cuda()
   

    framework.train(model=model,
                    model_name=prefix,
                    opt=opt,
                    save_ckpt=ckpt,
                    warmup_step=opt.warmup_step)
 
    # test
    res = framework.eval(model, ckpt=ckpt, L=opt.L)
    if not os.path.exists('./results'):
        os.mkdir('results')
    if  opt.dataset == 'snips':
        result_path = 'results/{}_{}_K{}_result.txt'.format(opt.dataset, opt.mode, opt.K)
    else:
        result_path = 'results/{}_{}_N{}_K{}_result.txt'.format(opt.dataset, opt.mode, opt.N, opt.K)
    with open(result_path, 'a') as f:
        f.write(prefix + '\n')
        f.write("{}\n".format(res))
    os.system(f'rm {ckpt}')

if __name__ == "__main__":
    main()
