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

from custom_dataloader import get_loader, get_label, set_multi_label
from sklearn.utils import resample
import torchvision
from Learner import Learner

CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'IMG_SIZE':244,
    'EPOCHS': 20,
    'LEARNING_RATE' : 0.001,
    'BATCH_SIZE' : 4,
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
        videotransforms.ReSize(256),
        videotransforms.RandomCrop(CFG['IMG_SIZE']),
        
        videotransforms.RandomHorizontalFlip(),
        videotransforms.ToTensor(),
        # videotransforms.Random_Resized_crop(size = CFG['IMG_SIZE'], scale=(0.4, 0.5)),
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
    valid_dataset, valid_loader = get_loader(data = val.video_path.values, label = val.label.values, transforms= test_transforms, batch_size=2)
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

    # i3d.replace_logits()      
    # i3d.replace_logits(157)
    # i3d.load_state_dict(torch.load('/home/cho092871/Documents/ckpt/dacon_img_300_resize_224/best.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    # optimizer = optim.SGD(i3d.parameters(), lr=CFG['LEARNING_RATE'])
    # optimizer = optim.SGD(i3d.parameters(), lr=CFG['LEARNING_RATE'], momentum=0.9, weight_decay=0.0000001)
    # lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 100, 300], gamma=0.1)
    optimizer = torch.optim.Adam(params = i3d.parameters(), lr = CFG["LEARNING_RATE"])
    # lr_sched = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=15, step_size_down=None, mode='triangular2', cycle_momentum=True)
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
    criterion = nn.CrossEntropyLoss().cuda()

    # criterion = losses.focal_loss(
    #     alpha = None,
    #     gamma = 5,
    #     reduction = 'mean',
    #     ignore_index = -100,
    #     device='cuda',
    #     dtype=torch.float32)

    
    best_score = 0
    num_steps_per_update = 1 # accum gradient
    steps = 0
    # train it
    for data_type in ['crash', 'ego', 'weather', 'time']:
        
        for epoch in range(CFG['EPOCHS']):
            train_carshtot = []
                    # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    i3d.train(True)
                else:
                    i3d.train(False)  # Set model to evaluate mode
                    
                tot_loss = []
                num_iter = 0
                optimizer.zero_grad()
                preds = []
                trues = []
    
                # Iterate over data.
                for data in tqdm.tqdm(dataloaders[phase]):
                    num_iter += 1
                    # get the inputs
                    inputs, labels = data
                    if data_type != 'crash':
                        labels = labels[labels !=0]
                        inputs = inputs[labels !=0]
                        
                    # print(labels)
                    labels = torch.tensor([set_multi_label(label)[data_type] for label in labels])
                    # print(labels)
                    # wrap them in Variable
                    inputs=inputs.cuda()
                    t = inputs.size(2)
                    labels=labels.cuda()
                    # print(labels)
                    # print(inputs.shape)
                    # label_crash, label_ego, label_weather, label_time  = Variable(crash.cuda()), Variable(ego.cuda()),Variable(weather.cuda()), Variable(time.cuda())
                    
                    # # with torch.cuda.amp.autocast(enabled=True):
                    out = i3d(inputs)
                    


    


if __name__ == '__main__':
    wandb.login()
    wandb.init(project="dacon")
    
    if not os.path.isdir(os.path.join("/home/cho/Documents/ckpt", args.save_model)):
        os.makedirs(os.path.join("/home/cho/Documents/ckpt", args.save_model))
        
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model)
