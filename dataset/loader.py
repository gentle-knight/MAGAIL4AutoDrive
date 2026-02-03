"""
统一数据加载：BC/MAGAIL 训练用专家 pkl 的加载函数与 Dataset。
主训练流水线使用本模块；dataset/expert_dataset.py 为可选 107 维/5 维管线。
"""
import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def load_expert_pkl(expert_data_path):
    """从目录或单个 pkl 加载专家 (obs, acts)，返回 concat 后的 obs_data, act_data。"""
    if os.path.isdir(expert_data_path):
        pkl_files = glob.glob(os.path.join(expert_data_path, "*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files in {expert_data_path}")
        print(f"Found {len(pkl_files)} pickle files in {expert_data_path}")
    elif os.path.exists(expert_data_path):
        pkl_files = [expert_data_path]
    else:
        raise FileNotFoundError(f"Expert data path not found: {expert_data_path}")

    obs_data, act_data = [], []
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                for traj in data:
                    if "obs" in traj and "acts" in traj:
                        obs_data.append(traj["obs"])
                        act_data.append(traj["acts"])
            elif isinstance(data, dict):
                if "observations" in data and "actions" in data:
                    obs_data.append(data["observations"])
                    act_data.append(data["actions"])
            else:
                print(f"Skipping {pkl_file}: Unknown data format {type(data)}")
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")

    if len(obs_data) == 0:
        raise ValueError("No valid data loaded from provided path.")
    obs_data = np.concatenate(obs_data, axis=0)
    act_data = np.concatenate(act_data, axis=0)
    print(f"Total loaded samples: {len(obs_data)}")
    return obs_data, act_data


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
        self.flat_data = []  # (obs, act) pairs

        # Load all .pkl files
        pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        print(f"Loading data from {len(pkl_files)} files in {data_dir}...")

        for pkl_file in pkl_files:
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                    # data is a list of dicts: {'obs': (T, 45), 'acts': (T, 2), ...}
                    self.trajectories.extend(data)
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")

        # Flatten for training Discriminator/BC
        print(f"Processing {len(self.trajectories)} trajectories...")
        for traj in self.trajectories:
            obs = traj["obs"]
            acts = traj["acts"]

            # obs: (T, 45), acts: (T, 2)
            for i in range(len(obs)):
                self.flat_data.append((obs[i], acts[i]))

        print(f"Total samples: {len(self.flat_data)}")

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        obs, act = self.flat_data[idx]

        obs = torch.from_numpy(obs).float()
        act = torch.from_numpy(act).float()

        sample = {"state": obs, "action": act}

        if self.transform:
            sample = self.transform(sample)

        return sample
