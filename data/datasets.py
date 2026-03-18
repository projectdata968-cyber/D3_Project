import os
import re
import pandas as pd
import albumentations as A
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

def get_number_from_filename(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def set_preprocessing():
    # Adding a slight center crop can help remove black bars or 
    # static edge UI which artificially lowers volatility scores.
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def read_video(folder_path, trans, num_frames=16, sampling_mode='consecutive'):
    image_paths = sorted(os.listdir(folder_path), key=get_number_from_filename)
    total_frames = len(image_paths)

    if total_frames < num_frames:
        # Fallback if video is too short: use linspace but repeat frames if necessary
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    elif sampling_mode == 'consecutive':
        # Select a random starting point and take a continuous block
        # This is CRITICAL for measuring real physical acceleration
        start_idx = random.randint(0, total_frames - num_frames)
        indices = np.arange(start_idx, start_idx + num_frames)
    else:
        # Original even sampling
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for idx in indices:
        image_path = os.path.join(folder_path, image_paths[idx])
        image = cv2.imread(image_path)
        if image is None: continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = trans(image=image)
        image = augmented["image"].transpose(2, 0, 1) # HWC -> CHW
        frames.append(image)

    # Ensure we always return exactly num_frames
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])

    return torch.from_numpy(np.stack(frames)).float()

class D3_dataset_AP(Dataset):
    def __init__(self, real_csv, fake_csv, max_len=9999999, mode='train'):
        super(D3_dataset_AP, self).__init__()
        df_real = pd.read_csv(real_csv).head(max_len)
        df_fake = pd.read_csv(fake_csv).head(max_len)
        self.df = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
        self.trans = set_preprocessing()
        self.mode = mode # 'train' uses random consecutive, 'eval' can use fixed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.loc[index]['label']
        frame_path = self.df.loc[index]['content_path']
        video_name = os.path.basename(frame_path)

        # Use consecutive sampling for better physics detection
        frames = read_video(frame_path, self.trans, sampling_mode='consecutive')
        return frames, label, video_name
