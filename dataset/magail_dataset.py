import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
import glob

class MAGAILExpertDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory containing .pkl files from generate_expert_data.py
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.trajectories = []
        self.flat_data = [] # (obs, act) pairs
        
        # Load all .pkl files
        pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        print(f"Loading data from {len(pkl_files)} files in {data_dir}...")
        
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    # data is a list of dicts: {'obs': (T, 45), 'acts': (T, 2), ...}
                    self.trajectories.extend(data)
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                
        # Flatten for training Discriminator/BC
        print(f"Processing {len(self.trajectories)} trajectories...")
        for traj in self.trajectories:
            obs = traj['obs']
            acts = traj['acts']
            
            # obs: (T, 45), acts: (T, 2)
            # We pair them up
            for i in range(len(obs)):
                self.flat_data.append((obs[i], acts[i]))
                
        print(f"Total samples: {len(self.flat_data)}")
        
    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        obs, act = self.flat_data[idx]
        
        # Convert to tensor
        obs = torch.from_numpy(obs).float()
        act = torch.from_numpy(act).float()
        
        sample = {'state': obs, 'action': act}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
