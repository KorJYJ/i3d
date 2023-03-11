import cv2
import numpy as np
from torchvision.datasets.video_utils import VideoClips
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from torchvision import datasets, transforms
import videotransforms
import torch
import cv2
from imblearn.over_sampling import SMOTE
from collections import Counter

root = "/home/cho/Documents/dacon/datasets"

def set_multi_label(label):
    """
        weather : [Normal, Snowy, Rainy]
        day = [Day, Night]
    """    
    if label ==0 :
        crash = 0
        ego = -1
        weather = -1
        time = -1
    elif label == 1:
        crash =1 
        ego = 1
        weather = 0
        time = 0
    elif label == 2:
        crash = 1
        ego = 1
        weather = 0
        time = 1
    elif label == 3:
        crash = 1
        ego = 1
        weather = 1
        time = 0
    elif label == 4:
        crash = 1
        ego = 1
        weather = 1
        time = 1
    elif label == 5:
        crash = 1
        ego = 1
        weather = 2
        time = 0
    elif label== 6:
        crash = 1
        ego = 1
        weather = 2
        time = 1
    elif label == 7:
        crash =1 
        ego = 0
        weather = 0
        time = 0
    elif label == 8:
        crash = 1
        ego = 0
        weather = 0
        time = 1
    elif label == 9:
        crash = 1
        ego = 1
        weather = 1
        time = 0
    elif label == 10:
        crash = 1
        ego = 0
        weather = 1
        time = 1
    elif label == 11:
        crash = 1
        ego = 0
        weather = 2
        time = 0
    elif label== 12:
        crash = 1
        ego = 0
        weather = 2
        time = 1
    return {'crash' : crash,
            'ego' : ego,
            'weather' : weather,
            'time' : time}
    

def get_label(multi_label):
    if multi_label == [0, None, None, None]:
        return 0 
    elif multi_label == [1, 1, 0, 0]:
        return 1
    elif multi_label == [1, 1, 0, 1]:
        return 2
    elif multi_label == [1, 1, 1, 0]:
        return 3
    elif multi_label == [1,1,1,1]:
        return 4
    elif multi_label == [1,1, 2,0]:
        return 5
    elif multi_label == [1, 1, 2, 1]:
        return 6
    elif multi_label == [1, 0, 0, 0]:
        return 7
    elif multi_label == [1, 0, 0, 1]:
        return 8
    elif multi_label == [1, 0, 1, 0]:
        return 9
    elif multi_label == [1, 0, 1, 1]:
        return 10
    elif multi_label == [1, 0, 2, 0]:
        return 11
    elif multi_label == [1, 0, 2, 1]:
        return 12

class DaconDataset(Dataset):
    def __init__(self, video_paths, labels = None, transform = None):
        self.dataset = VideoClips(
            video_paths = video_paths,
            clip_length_in_frames=50,
            frames_between_clips = 1,
            frame_rate =10,
            num_workers = 8,
        )
        
        self.transform = transform
        self.labels = labels
    
    def __getitem__(self, index):
        video, audio, info, video_idx = self.dataset.get_clip(index)
        video = video/255.
        
        # print(video.shape)
        if self.transform:
            video = video.numpy()
            video = self.transform(video)
            
        # if not self.labels:
        #     return video.permute(3, 0, 1, 2)
            
        # print(video.shape)
        label = self.labels[index]

        # label = set_multi_label(label)

        return video, label
        # return video
    
    def __len__(self):
        return self.dataset.num_clips()

def get_loader(data, label = None, transforms = None, batch_size=8):
    
        
    dataset = DaconDataset(data, label, transforms)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers=8
    )
    
    return dataset, dataloader
from sklearn.utils import resample

def main():
    train_transforms = transforms.Compose([
        videotransforms.RGB(),
        # videotransforms.ReSize(300),
        # videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
        videotransforms.GaussianBlurVideo(sigma_min=[0.0, 0.2], sigma_max=[0.0, 2.0])
    ])
    root = '/home/cho092871/Documents/dataset'
    df = pd.read_csv(os.path.join(root, 'train.csv'))
    train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, shuffle=False) # , random_state=cfg['SEED']
    
    print(train)
    print(Counter(train['label']))
    print(Counter(val['label']))
    data_path = []

    for v_p in train['video_path'].values:
        data_path.append(os.path.join(root, v_p.replace("./", "")))
    train['video_path'] = data_path
    
    data_path = []
    for v_p in val['video_path'].values:
        data_path.append(os.path.join(root, v_p.replace("./", "")))
    val['video_path'] = data_path
    
    upsample = [train[train.label==0], train[train.label==1], train[train.label==7]]
    for i in range(13):
        if i in [0, 7, 1]:
            continue
        data_len = len(train[train.label == i])
        upsample.append(resample(train[train.label == i],
                                    replace = True,
                                    n_samples = int(data_len * 255/(data_len * 3)),
                                    random_state=42))
    
    data = pd.concat(upsample)

    train_datatset, train_loader = get_loader(data = train.video_path.values, label = train.label.values, transforms = train_transforms)
    # print('h')
    # a = np.zeros((300, 300))
    # cv2.imshow('test', a)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


   
    idx = 0
    for idx, video in enumerate(train_datatset):
        out = cv2.VideoWriter(f'output_{idx}.mp4', fourcc, 10, (224, 224))
        for img in video:
            print(img.shape)
            img = img.numpy() * 255
            out.write(img.astype(np.uint8))
            
        out.release()

        if idx > 10:
            break
    
                

if __name__ == '__main__':
    main()