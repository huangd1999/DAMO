import argparse
import logging
from tqdm import tqdm
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dataset import Dataset
from DAMO import DAMO, RoutingModule
from test import test
from torchsummary import summary
import torchattacks
from wide_resnet import WideResNet

logger = logging.getLogger(__name__)

def random_inject(data,eps):
    data = data + torch.empty_like(data).uniform_(-eps, eps)
    data = torch.clamp(data, min=0, max=1).detach()
    return data


def self_attn_feature_alignment_loss(clean_feature, adv_feature):
    d = clean_feature.shape[1]
    attn = torch.matmul(clean_feature.flatten(start_dim=2), adv_feature.flatten(start_dim=2).permute(0,2,1))
    attn = F.softmax(attn / np.sqrt(d), dim=-1)

    common_feature = torch.matmul(attn, adv_feature.flatten(start_dim=2))
    print(common_feature.shape)
    feature_alignment_loss = F.mse_loss(common_feature, adv_feature)

    return feature_alignment_loss



def train(model, train_data, test_data, args):

    train_loader = train_data.get_dataloader(args.batch_size, shuffle=True)
    
    if args.loss == 'routing':
        routing_module = RoutingModule().cuda()
        optimizer = torch.optim.Adam(routing_module.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [7], gamma = 0.1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100,150], gamma = 0.1)
        criterion_kl = nn.KLDivLoss(size_average=False)
    device = next(model.parameters()).device
    
    best_acc = 0
    logger.info('Epoch \t Test acc \t Train Loss \t Train Acc')
    for epoch in range(1,args.epoch+1):
        # beta = 5 if epoch<45 else 1
        beta = 5
        train_loss = 0
        train_acc = 0
        train_n = 0
        model = model.train()
        with tqdm(train_loader) as loader:
            for data, label in loader:
                loader.set_description(f"Epoch {epoch}")
                
                optimizer.zero_grad()
                data, label= data.to(device), label.to(device)

                cleanoutputs = model(data)
                if args.attack == 'PGD':
                    attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=args.step, random_start=True)
                    data = attack(data, label)

                if args.loss == 'CE':
                    # Training Multi-branch model with Cross-Entrophy Loss
                    robustoutputs = model(data)
                    loss_robust = F.cross_entropy(robustoutputs, label)
                    cosine_loss = model.cosine_loss(data,args.branch)
                    loss = loss_robust  + beta * cosine_loss

                    acc = 1.0 - torch.count_nonzero(robustoutputs.argmax(dim=-1) - label) / data.shape[0]

                elif args.loss == 'trades':
                    # Training Multi-branch model with TRADES Loss
                    robustoutputs = model(data, args.branch)
                    loss_natural = F.cross_entropy(cleanoutputs, label)
                    loss_robust = (1.0 / args.batch_size) * criterion_kl(F.log_softmax(robustoutputs, dim=1),
                                                    F.softmax(cleanoutputs, dim=1))
                    cosine_loss = model.cosine_loss(data,args.branch)
                    loss = loss_natural  + beta * (loss_robust + cosine_loss)

                    acc = 1.0 - torch.count_nonzero(robustoutputs.argmax(dim=-1) - label) / data.shape[0]

                elif args.loss == 'routing':
                    # Training routing module with pseudo labels
                    robustoutputs = torch.concat([model(data,i).unsqueeze(1) for i in range(4)], dim=1)
                    branch = routing_module(data, args.branch)
       
                    pseudo_label = F.softmax(robustoutputs, dim=1)
                    pseudo_label = torch.zeros(label.shape[0]).cuda()
                    for i in range(label.shape[0]):
                        pseudo_label[i] = robustoutputs[i,:, label[i]].argmax()

                    loss = F.cross_entropy(branch, pseudo_label.long())

                    acc = 1.0 - torch.count_nonzero(branch.argmax(dim=-1) - pseudo_label) / data.shape[0]

                    
                # Backward and optimize
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * label.size(0)                
                train_acc += acc * label.size(0)
                train_n += label.shape[0]
                loader.set_postfix(loss=loss.item(), accuracy='{:.3f}'.format(acc))

        test_acc = test(model, test_data, args, mode='train')
        logger.info('%d \t %.4f \t %.4f \t %.4f',
            epoch, test_acc, train_loss/train_n, train_acc/train_n)

        scheduler.step()

        if epoch%10==0:
            acc_ori, acc_adv = test(model, test_data, args, mode='test')
            if acc_ori + acc_adv > best_acc:
                best_acc = acc_adv + acc_ori
                saved_name = '{0}-{1}-{2}.pt'.format('DAMO-ResNet20-branch', args.branch, args.dataset)
                torch.save(model.state_dict(), os.path.join(args.save_path, saved_name))
            logger.info('Test Acc \t PGD Acc')
            logger.info(' %.4f \t %.4f ', acc_ori, acc_adv)

    if args.loss == 'routing':
        torch.save(routing_module, './checkpoint/routing_module.pt')
    
        
        
        


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Adversarial Test')
    parser.add_argument('--save_path', type=str, default='./checkpoint', help='saved weight path')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
    parser.add_argument('--data_root', type=str, default='../data/', help='dataset path')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('--attack', type=str, default='PGD', help='attack method')
    parser.add_argument('--epoch', type=int, default=50, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--subnet', type=str, default='WRN', help='subnet')
    parser.add_argument('--out_dir', type=str, default='./log/', help='log output dir')
    parser.add_argument('--branch', type=int, default=0, help='training branch')
    parser.add_argument('--step', type=int, default=10, help='attack method') 
    parser.add_argument('--loss', type=str, default='trades', help='attack method') 
    parser.add_argument('--n_classes', type=int, default=10, help='num classes')
    args = parser.parse_args()


    logfile = os.path.join(args.out_dir, 'DAMO-ResNet20-branch{0}-{1}.log'.format(str(args.branch), args.dataset))
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = Dataset(path = args.data_root, dataset = args.dataset, train = True)
    test_data = Dataset(path = args.data_root, dataset = args.dataset, train = False)

    model = DAMO(args.subnet, num_classes=args.n_classes).cuda()
    # model = WideResNet().cuda()
    if args.pretrained != None:
        model.load_state_dict(torch.load(args.pretrained), strict=True)
    # print(model)
    summary(model, (3,32,32))
    if args.branch!=0:
        for para in model.head.parameters():
            para.requires_grad = False
    # for para in model.dynamic_layer[1].parameters():
    #             para.requires_grad = False
        for i in range(args.branch):
            for para in model.dynamic_layer[i].parameters():
                para.requires_grad = False
    train(model, train_data, test_data, args)
