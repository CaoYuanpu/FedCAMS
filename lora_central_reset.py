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
from utils import get_model, get_dataset, exp_details, average_parameter_delta
import loralib as lora
from torch.utils.data import DataLoader, Dataset
if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # define paths
#     out_dir_name = args.model + args.dataset + args.optimizer + '_lr' + str(args.lr) + '_locallr' + str(args.local_lr) + '_localep' + str(args.local_ep) +'_localbs' + str(args.local_bs) + '_eps' + str(args.eps)
    if 'lora' in args.model:
        file_name = '/{}_{}_{}_llr[{}]_bs[{}]_central_reset{}.pkl'.\
                    format(args.dataset, args.model, args.optimizer, 
                        args.local_lr, args.lr, args.reset)

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
    global_model = get_model(args.model, args.dataset, train_dataset[0][0].shape, num_classes, r=args.r)
    global_model.to(device)
    if 'lora' in args.model:
        lora.mark_only_lora_as_trainable(global_model, bias='all')

    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # global_model.train()
    # global_model.layer_input.train()
    # global_model.layer_hidden.train()
    

    # optimizer = torch.optim.SGD(global_model.parameters(), lr=args.local_lr, momentum=0)
    criterion = nn.CrossEntropyLoss().to(device)
    train_loss, test_loss, test_accuracy = [], [], []
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.local_lr, momentum=0)

    for epoch in tqdm(range(args.epochs)):
        total = 0
        batch_loss = []
        # print("start train")
        global_model.train()
        # print(global_model.layer_input.merged)
        # print(global_model.layer_input.merge_weights)
        # print("finish train")
        # print(global_model.layer_input.weight)
        # print(global_model.layer_input.lora_A)
        # print(global_model.layer_input.lora_B)
        # input()
        
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            global_model.zero_grad()
            logits = global_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item() * len(labels))
            total += len(labels)

        train_loss.append(sum(batch_loss)/total)


        # Test inference after completion of training
        test_acc, test_ls = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        test_loss.append(test_ls)

        # print global training loss after every rounds
        print(f'Train Loss : {train_loss[-1]}')
        print(f'Test Loss : {test_loss[-1]}')
        print(f'Test Accuracy : {test_accuracy[-1]} \n')
        logger.add_scalar('train loss', train_loss[-1], epoch)
        logger.add_scalar('test loss', test_loss[-1], epoch)
        logger.add_scalar('test acc', test_accuracy[-1], epoch)

        if args.save:
            # Saving the objects train_loss and train_accuracy:
            with open(args.outfolder + file_name, 'wb') as f:
                pickle.dump([test_loss, test_accuracy], f)
        if (epoch+1) % args.reset == 0:
            print(f'epoch: {epoch+1} reset')
            nn.init.kaiming_uniform_(global_model.layer_input.lora_A, a=math.sqrt(5))
            nn.init.zeros_(global_model.layer_input.lora_B)
            nn.init.kaiming_uniform_(global_model.layer_hidden.lora_A, a=math.sqrt(5))
            nn.init.zeros_(global_model.layer_hidden.lora_B)

        # print(global_model.layer_input.lora_A)
        # print(global_model.layer_input.lora_B)
        # print(global_model.layer_input.weight)
        # print(global_model.layer_input.merged)
        # print(global_model.layer_input.merge_weights)
        # input()

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
