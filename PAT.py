import argparse
import logging
from tqdm import tqdm
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dataset import Dataset
from BORT import BORT
from test import test
from torchsummary import summary
import torchattacks
import random
random.seed(0)

logger = logging.getLogger(__name__)

def random_inject(data,eps):
    data = data + torch.empty_like(data).uniform_(-eps, eps)
    data = torch.clamp(data, min=0, max=1).detach()
    return data

def train(model, train_data, test_data, args):

    train_loader = train_data.get_dataloader(args.batch_size, shuffle=True)
    criterion_kl = nn.KLDivLoss(size_average=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40], gamma = 0.1)

    device = next(model.parameters()).device
    
    best_acc = 0
    logger.info('Epoch \t Test acc \t Train Loss \t Train Acc')
    for epoch in range(args.epoch):
        beta = 3 if epoch<40 else 1
        train_loss = 0
        train_acc = 0
        train_n = 0
        model = model.train()
        with tqdm(train_loader) as loader:
            for data, label in loader:
                loader.set_description(f"Epoch {epoch+1}")
                
                optimizer.zero_grad()
                data, label= data.to(device), label.to(device)
                #print(label)
                clear_outputs = model(data)
                loss_natural = F.cross_entropy(clear_outputs, label)
                if args.attack == 'PGD':
                    attack = torchattacks.PGD(model, eps=8/255, alpha=0.01, steps=10, random_start=True)
                    data = attack(data, label)
                outputs = model(data)
                # loss = criterion(outputs, label)
                # cosine_loss = model.cosine_loss(data,args.branch)
                loss_robust = (1.0 / args.batch_size) * criterion_kl(F.log_softmax(outputs, dim=1),
                                                    F.softmax(clear_outputs, dim=1))
                loss = loss_natural + beta * loss_robust #+ beta * cosine_loss
                
                # Backward and optimize
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * label.size(0)
                acc = 1.0 - torch.count_nonzero(outputs.argmax(dim=-1) - label) / data.shape[0]
                train_acc += acc * label.size(0)
                train_n += label.shape[0]
                loader.set_postfix(loss=loss.item(), accuracy='{:.3f}'.format(acc))

        test_acc = test(model, test_data, args, mode='train')

        logger.info('%d \t %.4f \t %.4f \t %.4f',
            epoch, test_acc, train_loss/train_n, train_acc/train_n)
        if test_acc > best_acc:
            best_acc = test_acc
            saved_name = '{0}-{1}-{2}.pt'.format('tradesWRN-28-10', args.dataset, args.attack)
            torch.save(model.state_dict(), os.path.join(args.save_path, saved_name))
        scheduler.step()
        if epoch%10==0:
            acc_ori, acc_adv = test(model, test_data, args, mode='test')

            logger.info('Test Acc \t PGD Acc')
            logger.info(' %.4f \t %.4f ', acc_ori, acc_adv)
        
        
        


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
    parser.add_argument('--out_dir', type=str, default='./train_log/', help='log output dir')
    parser.add_argument('--branch', type=int, default=2, help='training branch')
    parser.add_argument('--step', type=int, default=40, help='attack method') 
    parser.add_argument('--n_classes', type=int, default=10, help='num classes')
    args = parser.parse_args()

    logfile = os.path.join(args.out_dir, 'trades-WRN28-10-branch{0}-{1}.log'.format(str(args.branch), args.dataset))
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

    model = BORT(args.subnet, num_classes=args.n_classes).cuda()
    if args.pretrained != None:
        model.load_state_dict(torch.load(args.pretrained), strict=True)
    # print(model)
    summary(model, (3,32,32))
    train(model, train_data, test_data, args)
