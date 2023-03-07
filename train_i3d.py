import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='rgb or flow')
parser.add_argument('--save_model', type=str)
parser.add_argument('--root', type=str)

args = parser.parse_args()

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset import CustomDataset as Dataset

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import wandb


from custom_dataloader import get_loader

CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'IMG_SIZE':224,
    'EPOCHS':50,
    'LEARNING_RATE':3e-3,
    'BATCH_SIZE' : 3,
    'SEED':41
}


def run(init_lr=0.0001, max_steps=64e3, mode='rgb', root='/home/cho/Documents/dacon/datasets', save_model=''):
    # setup dataset
    # train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
    #                                        videotransforms.RandomHorizontalFlip(),
    # ])
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # df = pd.read_csv(os.path.join(root, 'train.csv'))

    CFG['ROOT_PATH'] = root
    
    # train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])

    # dataset = Dataset(train['video_path'].values, train['label'].values, CFG, train_transforms)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    # val_dataset = Dataset(val['video_path'].values, val['label'].values, CFG, test_transforms)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    df = pd.read_csv(os.path.join(root, 'train.csv'))
    train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED']) # , random_state=cfg['SEED']

    train_transforms = transforms.Compose([
        videotransforms.RGB(),
        videotransforms.ReSize(300),
        videotransforms.RandomCrop(CFG['IMG_SIZE']),
        videotransforms.RandomHorizontalFlip(),
    ])
    
    test_transforms = transforms.Compose([
        videotransforms.RGB(),
        videotransforms.ReSize(CFG['IMG_SIZE']),
    ])

    train_datatset, train_loader = get_loader(data = train, root = root, transforms = train_transforms, batch_size=CFG['BATCH_SIZE'])
    valid_dataset, valid_loader = get_loader(data = val, root = root, transforms= test_transforms, batch_size=CFG['BATCH_SIZE'])
    # video_path_list, label_list, cfg, transforms = None
    # dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    # val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)    

    dataloaders = {'train': train_loader, 'val': valid_loader}
    datasets = {'train': train_datatset, 'val': valid_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(13, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

    i3d.replace_logits(13)
    # i3d.load_state_dict(torch.load('dacon_1003140.pt'))
        
    
    # i3d.replace_logits(157)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    optimizer = optim.Adam(i3d.parameters(), lr=CFG['LEARNING_RATE'])
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    # lr_sched = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=50, step_size_down=None, mode='triangular2')

    criterion = nn.CrossEntropyLoss().cuda()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    num_steps_per_update = 1 # accum gradient
    steps = 0
    # train it
    for epoch in range(CFG['EPOCHS']):
        train_tot = []
        valid_tot = []
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            trues = []
            preds = []
            # Iterate over data.
            for data in tqdm.tqdm(dataloaders[phase]):
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())
                    
                # with torch.cuda.amp.autocast(enabled=True):
                output = i3d(inputs)
                
                loss = criterion(output, labels)
                if phase == 'train':
                    wandb.log({'Train iter Loss': loss})
                # upsample to input size
                # per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # # compute localization loss
                # loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                # tot_loc_loss += loc_loss.data[0]

                # # compute classification loss (with max-pooling along time B x C x T)
                # cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                # tot_cls_loss += cls_loss.data[0]

                # loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.item() 
                if phase == 'train':
                    train_tot.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                elif phase == 'val':
                    valid_tot.append(loss.item())
                    
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                # lr_sched.step()

                preds += output.argmax(1).detach().cpu().numpy().tolist()
                trues += labels.detach().cpu().numpy().tolist()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    if steps % 10 == 0:
                        train_f1_score = f1_score(trues, preds, average='macro')
                        print('{} Loss: {:.4f}, F1 score: {:.4f}'.format(phase, tot_loss/10, train_f1_score))
                        wandb.log({'f1 score' : train_f1_score})
                        # save model
                        torch.save(i3d.module.state_dict(), os.path.join("/home/cho/Documents/dacon/pytorch-i3d/ckpt", f"{save_model}_{str(steps).zfill(6)}"+'.pt'))
                        tot_loss = 0.
            
            if phase == 'val':
                _val_score = f1_score(trues, preds, average='macro')
                print('{} Train Loss: {:.4f}  Valid Loss: {:.4f} Valid F1 score: {:.4f}'.format(phase, np.mean(train_tot), np.mean(valid_tot), _val_score ))
                wandb.log({'Train Loss': np.mean(train_tot), 'Valid Loss': np.mean(valid_tot), "Valid F1 score" : _val_score})


    


if __name__ == '__main__':
    wandb.login()
    wandb.init(project="my-awesome-project")
    
    if not os.path.isdir(os.path.join('ckpt', args.save_model)):
        os.makedirs(os.path.join('ckpt', args.save_model))
        
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model)
