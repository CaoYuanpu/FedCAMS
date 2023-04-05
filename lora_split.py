#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import math
import torch
from torch import nn
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, update_model_inplace, test_inference
from utils import get_model, get_dataset, average_weights_lora, average_weights_lora_split, exp_details, average_parameter_delta
import loralib as lora

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # define paths
#     out_dir_name = args.model + args.dataset + args.optimizer + '_lr' + str(args.lr) + '_locallr' + str(args.local_lr) + '_localep' + str(args.local_ep) +'_localbs' + str(args.local_bs) + '_eps' + str(args.eps)
    file_name = '/{}_{}_{}_llr[{}]_glr[{}]_eps[{}]_le[{}]_bs[{}]_iid[{}]_mi[{}]_frac[{}]_split.pkl'.\
                format(args.dataset, args.model, args.optimizer, 
                    args.local_lr, args.lr, args.eps, 
                    args.local_ep, args.local_bs, args.iid, args.max_init, args.frac)
    logger = SummaryWriter('./logs/'+file_name)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1) # limit cpu use
    print ('-- pytorch version: ', torch.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.outfolder):
        os.mkdir(args.outfolder)

    # load dataset and user groups
    train_dataset, test_dataset, num_classes, user_groups = get_dataset(args)

    # Set the model to train and send it to device.
    global_model = get_model(args.model, args.dataset, train_dataset[0][0].shape, num_classes)
    global_model.to(device)
    lora.mark_only_lora_as_trainable(global_model, bias='all')


    # Training
    train_loss_sampled, train_loss, train_accuracy = [], [], []
    test_loss, test_accuracy = [], []
    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        ep_time = time.time()
        
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        global_model.train()

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            
            w, _, loss = local_model.update_weights_local(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        

        bn_weights = average_weights_lora_split(local_weights)
        global_model.load_state_dict(bn_weights, strict=False)

        # report and store loss and accuracy
        # this is local training loss on sampled users
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        print('Epoch Run Time: {0:0.4f} of {1} global rounds'.format(time.time()-ep_time, epoch+1))
        print(f'Training Loss : {train_loss[-1]}')
        logger.add_scalar('train loss', train_loss[-1], epoch)

         
        # Test inference after completion of training
        test_acc, test_ls = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        test_loss.append(test_ls)

        # print global training loss after every rounds

        print(f'Test Loss : {test_loss[-1]}')
        print(f'Test Accuracy : {test_accuracy[-1]} \n')

        logger.add_scalar('test loss', test_loss[-1], epoch)
        logger.add_scalar('test acc', test_accuracy[-1], epoch)

        if args.save:
            # Saving the objects train_loss and train_accuracy:
            with open(args.outfolder + file_name, 'wb') as f:
                pickle.dump([train_loss, test_loss, test_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
