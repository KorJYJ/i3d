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
from sklearn.metrics import f1_score, accuracy_score
import wandb

from utils import losses
from custom_dataloader import get_loader, get_label
from sklearn.utils import resample
import torchvision
CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'IMG_SIZE':244,
    'EPOCHS': 400,
    'LEARNING_RATE' : 0.001,
    'BATCH_SIZE' : 24,
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
    train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'], stratify=df['label']) # , random_state=cfg['SEED']
    #720, 1280
    train_transforms = transforms.Compose([
        videotransforms.RGB(),
        # videotransforms.ReSize(256),
        # videotransforms.RandomCrop(CFG['IMG_SIZE']),
        
        videotransforms.RandomHorizontalFlip(),
        videotransforms.ToTensor(),
        videotransforms.Random_Resized_crop(size = CFG['IMG_SIZE'], scale=(0.4, 0.5)),
        # videotransforms.GaussianBlurVideo()
    ])
    
    test_transforms = transforms.Compose([
        videotransforms.RGB(),
        videotransforms.ReSize(244),
        videotransforms.ToTensor()

        # videotransforms.CenterCrop(CFG['IMG_SIZE']),
    ])
    
    data_path = []

    for v_p in train['video_path'].values:
        data_path.append(os.path.join(root, v_p.replace("./", "")))
    train['video_path'] = data_path
    
    data_path = []
    for v_p in val['video_path'].values:
        data_path.append(os.path.join(root, v_p.replace("./", "")))
    val['video_path'] = data_path
    
    # upsample = [train[train.label==0], train[train.label==1], train[train.label==7]]
    # for i in range(13):
    #     if i in [0, 7, 1]:
    #         continue
    #     data_len = len(train[train.label == i])
    #     upsample.append(resample(train[train.label == i],
    #                                 replace = True,
    #                                 n_samples = int(data_len * 255/(data_len * 3)),
    #                                 random_state=42))
    
    # data = pd.concat(upsample)

    train_datatset, train_loader = get_loader(data = train.video_path.values, label = train.label.values, transforms = train_transforms, batch_size=CFG['BATCH_SIZE'])
    valid_dataset, valid_loader = get_loader(data = val.video_path.values, label = val.label.values, transforms= test_transforms, batch_size=CFG['BATCH_SIZE'])
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
    
    # i3d.replace_logits(157)
    i3d.load_state_dict(torch.load('/home/cho092871/Documents/ckpt/dacon_img_300_resize_224/best.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    # optimizer = optim.SGD(i3d.parameters(), lr=CFG['LEARNING_RATE'])
    # optimizer = optim.SGD(i3d.parameters(), lr=CFG['LEARNING_RATE'], momentum=0.9, weight_decay=0.0000001)
    # lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 100, 300], gamma=0.1)
    optimizer = torch.optim.Adam(params = i3d.parameters(), lr = CFG["LEARNING_RATE"])
    # lr_sched = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=15, step_size_down=None, mode='triangular2', cycle_momentum=True)
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
    criterion = nn.CrossEntropyLoss().cuda()

    criterion = losses.focal_loss(
        alpha = None,
        gamma = 5,
        reduction = 'mean',
        ignore_index = -100,
        device='cuda',
        dtype=torch.float32)

    
    best_score = 0
    num_steps_per_update = 1 # accum gradient
    steps = 0
    # train it
    for epoch in range(CFG['EPOCHS']):
        train_carshtot = []
                # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_crash_loss = 0.0
            tot_ego_loss = 0.0
            tot_weather_loss = 0.0
            tot_time_loss = 0.0
            
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            preds = []
            trues = []
            trues_crash, trues_ego, trues_weather, trues_time = [], [], [], []
            preds_crash, preds_ego, preds_weather, preds_time = [], [], [], [] 
            # Iterate over data.
            for data in tqdm.tqdm(dataloaders[phase]):
                num_iter += 1
                # get the inputs
                inputs, crash, ego, weather, time = data
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                # labels = Variable(labels.cuda())
                label_crash, label_ego, label_weather, label_time  = Variable(crash.cuda()), Variable(ego.cuda()),Variable(weather.cuda()), Variable(time.cuda())
                
                # # with torch.cuda.amp.autocast(enabled=True):
                out_crash, out_ego, out_weather, out_time = i3d(inputs)
                # # upsample to input size
                # per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # # compute localization loss
                # loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                # tot_loc_loss += loc_loss.data[0]

                # # compute classification loss (with max-pooling along time B x C x T)
                # cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                # tot_cls_loss += cls_loss.data[0]

                # loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                loss_crash = criterion(out_crash, label_crash)
                loss_ego = criterion(out_ego, label_ego)
                loss_weather = criterion(out_weather, label_weather)
                loss_time = criterion(out_time, label_time)
                
                
                loss_ego[label_crash ==0] = 0
                loss_weather[label_crash ==0 ] = 0
                loss_time[label_crash ==0 ] = 0
                
                tot_crash_loss += loss_crash.item() 
                tot_ego_loss += loss_ego.item()
                tot_time_loss += loss_time.item()
                tot_weather_loss += loss_weather.itme()

                
                # loss = criterion(output, labels)
                if phase == 'train':
                    # wandb.log({'Train iter Loss': loss})
                    wandb.log({
                        'loss_ego' : loss_ego,
                        'loss_crash' : loss_crash,
                        'loss_weather' : loss_weather,
                        'loss_time' : loss_time})
                    loss_ego.backward()
                    loss_crash.backward()
                    loss_weather.backward()
                    loss_time.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                # lr_sched.step()

                pred_crash = out_crash.argmax(1).detach().cpu().numpy().tolist()
                pred_ego = out_ego.argmax(1).detach().cpu().numpy().tolist()
                pred_weather = out_weather.argmax(1).detach().cpu().numpy().tolist()
                pred_time = out_time.argmax(1).detach().cpu().numpy().tolist()
                
                for a in zip(pred_crash, pred_ego, pred_weather, pred_time):
                    if pred_crash[0] ==0:
                        preds += [0]
                    else :
                        preds += [get_label(a)]
                        
                preds_crash += pred_crash
                if pred_crash ==1:
                    preds_ego += pred_ego
                    preds_weather += pred_weather
                    preds_time += pred_time
                
                
                trues_crash += label_crash.detach().cpu().numpy().tolist()
                if pred_crash ==1:
                    trues_ego += label_ego.detach().cpu().numpy().tolist()
                    trues_weather += label_weather.detach().cpu().numpy().tolist()
                    trues_time += label_time.detach().cpu().numpy().tolist()
                
                if phase == 'train':
                    steps += 1
                    if steps % 10 == 0:            
                        train_acc = accuracy_score(trues, preds)
                        train_f1_score = f1_score(trues, preds, average='macro')
                        train_crash_acc = accuracy_score(trues, preds)
                        train_crash_f1_score = f1_score(trues, preds, average='macro')
                        train_ego_acc = accuracy_score(trues, preds)
                        train_f1_score = f1_score(trues, preds, average='macro')
                        train_acc = accuracy_score(trues, preds)
                        train_f1_score = f1_score(trues, preds, average='macro')
                        train_acc = accuracy_score(trues, preds)
                        train_f1_score = f1_score(trues, preds, average='macro')
                        print('{} Loss: {:.4f}, F1 score: {:.4f} Acc : {:.4f}'.format(phase, tot_loss/10, train_f1_score , train_acc))
                        wandb.log({'f1 score' : train_f1_score, 
                                    'acc' : train_acc})
                        tot_loss = 0.


            if phase == 'train':
                torch.save(i3d.module.state_dict(), os.path.join("/home/cho092871/Documents/ckpt", save_model,f"{save_model}_{str(epoch).zfill(6)}"+'.pt'))
            
            if phase == 'val':
                
                _val_score = f1_score(trues, preds, average='macro')
                if best_score < _val_score:
                    best_score = _val_score
                    torch.save(i3d.module.state_dict(), os.path.join("/home/cho092871/Documents/ckpt", save_model, "best.pt"))
                print('{} Train Loss: {:.4f}  Valid Loss: {:.4f} Valid F1 score: {:.4f}'.format(phase, np.mean(train_tot), np.mean(valid_tot), _val_score ))
                wandb.log({'Train Loss': np.mean(train_tot), 'Valid Loss': np.mean(valid_tot), "Valid F1 score" : _val_score})
        # lr_sched.step()
            


    


if __name__ == '__main__':
    wandb.login()
    wandb.init(project="dacon")
    
    if not os.path.isdir(os.path.join("/home/cho092871/Documents/ckpt", args.save_model)):
        os.makedirs(os.path.join("/home/cho092871/Documents/ckpt", args.save_model))
        
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model)
