import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchattacks
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.models as models
from ARC_attack import attack_ARC_linf
from dataset import Dataset
from tSNE import tsne
from DAMO import DAMO
from PGD import PGD


def test_(model, routing_module, test_data, args, mode='test'):
    if args.attack != None and mode == 'test':
        if args.attack == 'AutoAttack':
            attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        elif args.attack == 'APGD':
            attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=20, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
        elif args.attack == 'PGD':
            # attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=args.step, random_start=True)
            attack = PGD(model, eps=8/255, alpha=2/255, steps=args.step, random_start=True)
        elif args.attack == 'FGSM':
            attack = torchattacks.FGSM(model, eps=8/255)
        elif args.attack == 'CW':
            attack = torchattacks.CW(model)
        elif args.attack == 'Square':
            attack = torchattacks.Square(model, norm='Linf', eps=16/255, n_queries=10000)

    test_loader = test_data.get_dataloader(batch_size=args.batch_size, shuffle=True)
    correct = 0
    correct_adv = 0
    all_samples = 0
    device = next(model.parameters()).device
    model = model.eval()
    batch_id = 0

    with tqdm(test_loader) as test_loader:
        for data, label in test_loader:
            batch_id+=1

            data, label = data.to(device), label.to(device)

            outputs = model(data)
            correct += data.shape[0] - torch.count_nonzero(outputs.argmax(dim=-1) - label)
            if args.attack != None and mode == 'test':
                data = attack(data, label)
                correct_adv_list = []
                
                branch = routing_module(data)
                branch = round(branch.argmax(dim=1).sum().item() / label.shape[0])
                ''' ARC Evaluation '''
                # delta = attack_ARC_linf(model, data, label, 8/255, [0.9,0.1], 10, 2/255)
                # delta = delta.detach()
                # outputs = model(torch.clamp(data+delta, 0, 1), branch)
                outputs = model(data, branch)
                correct_adv += (label.shape[0] - torch.count_nonzero(outputs.argmax(dim=-1) - label)).item()


            all_samples += args.batch_size
            if args.attack == 'AutoAttack':
                print('batch_id: {} \t Attack: {} \t Accuracy = {}/{} = {:.3f}'.format(batch_id,args.attack, correct_adv, all_samples, correct_adv*1.0 / all_samples))
            

    acc_ori = correct*1.0 / all_samples
    print('Attack: None \t Accuracy = {}/{} = {:.3f}'.format(correct, all_samples, acc_ori))
    acc_adv = 0
    if args.attack != 'None' and mode == 'test':
        acc_adv = correct_adv*1.0 / all_samples
        print('Attack: {} \t Accuracy = {}/{} = {:.3f}'.format(args.attack, correct_adv, all_samples, acc_adv))
    
    if mode =='train':
        return acc_ori  
    else:
        return acc_ori, acc_adv


def test(model, test_data, args, mode='test'):
    if args.attack != None and mode == 'test':
        if args.attack == 'AutoAttack':
            attack = AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
            # data = attack(data, label)
        elif args.attack == 'APGD':
            attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
        elif args.attack == 'PGD':
            # attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=args.step, random_start=True)
            attack = PGD(model, eps=8/255, alpha=2/255, steps=args.step, random_start=True)
            # data = attack(data, label)
        elif args.attack == 'FGSM':
            attack = torchattacks.FGSM(model, eps=8/255)
        elif args.attack == 'CW':
            attack = torchattacks.CW(model)
        elif args.attack == 'Square':
            attack = torchattacks.Square(model, norm='Linf', eps=8/255, n_queries=10000)
    test_loader = test_data.get_dataloader(batch_size=args.batch_size, shuffle=True)
    correct = 0
    correct_adv = 0
    all_samples = 0
    device = next(model.parameters()).device
    model = model.eval()
    batch_id = 0
    with tqdm(test_loader) as test_loader:
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)

            outputs = model(data)
            correct += data.shape[0] - torch.count_nonzero(outputs.argmax(dim=-1) - label)

            if args.attack != None and mode == 'test':
                data = attack(data, label)
                # delta = attack_ARC_linf(model, data, label, 8/255, [0.8,0.2], 10, 2/255)
                # delta = delta.detach()
                # outputs = model(torch.clamp(data+delta, 0, 1))
                outputs = model(data)
                correct_adv += data.shape[0] - torch.count_nonzero(outputs.argmax(dim=-1) - label)
            all_samples += data.shape[0]

            if args.attack == 'AutoAttack':
                print('batch_id: {} \t Attack: {} \t Accuracy = {}/{} = {:.3f}'.format(batch_id,args.attack, correct_adv, all_samples, correct_adv*1.0 / all_samples))
            

    acc_ori = correct*1.0 / all_samples
    print('Attack: None \t Accuracy = {}/{} = {:.3f}'.format(correct, all_samples, acc_ori))
    acc_adv = 0
    if args.attack != 'None' and mode == 'test':
        acc_adv = correct_adv*1.0 / all_samples
        print('Attack: {} \t Accuracy = {}/{} = {:.3f}'.format(args.attack, correct_adv, all_samples, acc_adv))
    
    if mode =='train':
        return acc_ori  
    else:
        return acc_ori, acc_adv


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Adversarial Test')
    parser.add_argument('--weights', type=str, default='./checkpoint/DAMO-ResNet20-branch-2-CIFAR10.pt', help='saved model path')
    parser.add_argument('--routing_weights', type=str, default='./checkpoint/routing_WRN-34-10-cifar10.pt', help='saved routing module path')
    parser.add_argument('--data_root', type=str, default='../data/', help='dataset path')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size')
    parser.add_argument('--attack', type=str, default=None, help='attack method')
    parser.add_argument('--step', type=int, default=40, help='attack method')
    args = parser.parse_args()
    device = 'cuda'
    model = DAMO(sub_network='WRN').to(device)
    model.load_state_dict(torch.load(args.weights), strict=True)
    model = model.to(device)
    test_data = Dataset(path = args.data_root, dataset = args.dataset, train = False)

    routing_module = torch.load('./checkpoint/routing_WRN-34-10-cifar10.pt')
    # acc = test(model, test_data, args)
    acc = test_(model, routing_module, test_data, args)
