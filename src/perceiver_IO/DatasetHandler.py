import torch
import sys
import numpy as np
from os.path import exists
from torch.utils.data import Dataset, DataLoader

class PPDataset(Dataset):

    def __init__(self,name,data_path,inout_size=1024):
        self.data_path = data_path
        self.name = name
        self.inout_size=inout_size
        data_exist = exists(data_path)

        if not data_exist:
            print("Default preprocessed data does not exist..")
            sys.exit()

        print("Loading dataset..")
        
        self.geometry, self.charges, self.path, self.ordering = torch.load(self.data_path)
        print("Finished loading dataset")

    def __len__(self):
        return len(self.geometry)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        geometry = torch.zeros(self.inout_size, self.inout_size)
        charges = torch.zeros(self.inout_size)

        charge_ones = torch.ones(len(self.charges[idx]))


        charge_mask = torch.zeros(self.inout_size)
        charge_mask[:len(charge_ones)] = charge_ones

        geometry_ones = torch.ones(len(self.geometry[idx]),len(self.geometry[idx]))
        geometry_mask = torch.zeros(self.inout_size, self.inout_size)
        geometry_mask[:len(geometry_ones),:len(geometry_ones)] = geometry_ones

        geometry[:len(self.geometry[idx]),:len(self.geometry[idx])]=self.geometry[idx]
        charges[:len(self.charges[idx])] = torch.sub(self.charges[idx],4)*10
        sample = {'geometry': geometry, 'charges': charges,"geometry_mask":geometry_mask, "charges_mask":charge_mask, "ordering": self.ordering[idx], "path":self.path[idx] }

        return sample

class MeshDataset(Dataset):

    def __init__(self,name,data_path):
        self.data_path = data_path
        self.name = name
        data_exist = exists(data_path+"_features.npy")

        if not data_exist:
            print("Default preprocessed features data does not exist..")
            sys.exit()

        data_exist = exists(data_path+"_labels.npy")

        if not data_exist:
            print("Default preprocessed labels data does not exist..")
            sys.exit()

        print("Loading features dataset..")
        self.data_features= torch.from_numpy(np.load(self.data_path+"_features.npy"))
        print("Loading labels dataset..")
        self.data_labels= torch.from_numpy(np.load(self.data_path+"_labels.npy"))

        print("Finished loading dataset")

    def __len__(self):
        return len(self.data_features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'features': self.data_features[idx].unsqueeze(0).float(), 'labels': self.data_labels[idx].float()}

        return sample