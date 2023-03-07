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

root = "/home/cho/Documents/dacon/datasets"

class DaconDataset(Dataset):
    def __init__(self, video_paths, labels, transform = None):
        self.dataset = VideoClips(
            video_paths = video_paths,
            clip_length_in_frames=50.,
            frames_between_clips = 1.,
            frame_rate =10.,
            num_workers = 5,
        )
        
        self.transform = transform
        self.labels = labels
    
    def __getitem__(self, index):
        video, audio, info, video_idx = self.dataset.get_clip(index)
        video = video/255*2 -1.
        
        # print(video.shape)
        if self.transform:
            video = video.numpy()
            video = self.transform(video)
            video = torch.from_numpy(video)
            
        # print(video.shape)
        label = self.labels[index]
        return video.permute(3, 0, 1, 2), label
        # return video
    
    def __len__(self):
        return self.dataset.num_clips()

def get_loader(data, root, transforms = None, batch_size=8):
    data_path = []

    for v_p in data['video_path'].values:
        data_path.append(os.path.join(root, v_p.replace("./", "")))

    dataset = DaconDataset(data_path, data['label'].values, transforms)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers=10
    )
    
    return dataset, dataloader

def main():
    train_dataset, valid_dataset, train_dataloader, valid_dataloader = get_loader('/home/cho/Documents/dacon/datasets')
    # import time
    # print('h')
    # a = np.zeros((300, 300))
    # cv2.imshow('test', a)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


    out = cv2.VideoWriter('output.mp4', fourcc, 30, (224, 224))

    for video in train_dataset:
        for img in video:
            print(img.shape)
            img = img.numpy()
            out.write(img.astype(np.uint8))
        break
    
    out.release()
                

if __name__ == '__main__':
    main()