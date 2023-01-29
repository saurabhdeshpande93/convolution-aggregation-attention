import os
import torch
import pandas as pd
import numpy as np
import sys

class raw_db():

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.gen_paths = []
        self.charge_paths = []


        for (root,dirs,files) in os.walk(root_dir):
            for files in files:
                if files.endswith(".gen"):  
                    name = root+'/'+files
                    name = name[:-3]
                    self.gen_paths.append(name+"gen")
                    self.charge_paths.append(name+"dat")

    def __len__(self):
        return len(self.gen_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gen = pd.read_csv(self.gen_paths[idx],sep = " ",header=None,skiprows=2,usecols=[5,6,7])
        charges = pd.read_csv(self.charge_paths[idx],sep = " ",header=None,skiprows=2,usecols=[3],nrows=len(gen))


        sample = {'geometry': torch.tensor(gen.values).squeeze(), 'charges': torch.tensor(charges.values).squeeze(), 'path':self.gen_paths[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

def torchsave(name,raw_dataset):

    gen = []
    charges = []
    path = []
    ordering = []
    for i in range(0,int(len(raw_dataset)/10)):
        print("Transforming test dataset "+name+".. "+str(i)+"/"+str(int(len(raw_dataset)/10)))
        gen.append(raw_dataset[i]['geometry'])
        charges.append(raw_dataset[i]['charges'])
        path.append(raw_dataset[i]['path'])
        ordering.append(raw_dataset[i]['ordering'])
    print("Saving test dataset "+name+"..")
    torch.save([gen,charges,path,ordering], name+'_test.torch')
    print("Dataset "+name+"_test.torch saved..")

    gen = []
    charges = []
    path = []
    ordering = []
    for i in range(int(len(raw_dataset)/10),len(raw_dataset)):
        print("Transforming train dataset "+name+".. "+str(i)+"/"+str(len(raw_dataset)))
        gen.append(raw_dataset[i]['geometry'])
        charges.append(raw_dataset[i]['charges'])
        path.append(raw_dataset[i]['path'])
        ordering.append(raw_dataset[i]['ordering'])
    print("Saving train dataset "+name+"..")
    torch.save([gen,charges,path,ordering], name+'_train.torch')
    print("Dataset "+name+"_train.torch saved..")
    
def writecharges(charges,mask,ordering,name=None):

    Nat = int(mask.cpu().sum())
    charges = charges.cpu()

    inv_ordering=np.argsort(ordering.numpy())

    charges = charges.numpy()
    charges = charges[:Nat]
    charges = charges[inv_ordering]
    totalcharge = 0
    for i in range(Nat):
        totalcharge=totalcharge+charges[i]/10+4

    prop_factor=totalcharge/(4*Nat)

    if name == None:
        name = "charges"
    print("Saving charges for "+name)
    with open(name+".dat", 'w') as f:
        f.write("           6\n")
        f.write(" F F F T          "+str(Nat)+"           1   "+str(4*Nat)+"\n")
        for i in range(Nat):
            f.write("   "+str((charges[i]/10+4)/prop_factor)+"        0.0000000000000000        0.0000000000000000        0.0000000000000000     \n")
        f.write(" 0 0\n")

class model_state_handler():
    def __init__(self, name,root_dir="SavedModels"):
        self.name=name
        self.root_dir = root_dir
        self.states = []
        self.current_loss_str="10000000000000" #14 total digits
        self.it = 0

        print("Loading from checkpoint "+ self.name)
        for (root,dirs,files) in os.walk(root_dir):
            for files in files:
                if files.startswith(self.name) & files.endswith(".pt"):  
                    name = root+'/'+files
                    name = name[:-3]
                    self.states.append(name)

        if len(self.states) == 1:
            with open(self.root_dir+'/'+self.name+".txt", 'rb') as f:
                try:  # catch OSError in case of a one line file 
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)
                last_line = f.readline().decode()
            last_line = last_line.split()
            self.it = int(last_line[0])
            
        if len(self.states) == 0:
            print("No states found with name "+ self.name)
            self.states.append(self.root_dir+'/'+self.name+"_"+self.current_loss_str)
            print("Starting new state with default I.C. "+ self.states[0])
            self.current_loss_str=self.states[0][-14:]
            self.it=0

        if len(self.states) > 1:
            print("Multiple states found with name "+ self.name)
            for i in range(len(self.states)):
                print(self.states[i])
            sys.exit()



        self.current_loss_str=self.states[0][-14:]


    def Update(self,it,train_v,test_v):
        save_model=None
        if float(self.current_loss_str) > test_v:
            prev_model=self.states[0]
            aux_str="{:.13f}".format(test_v)
            aux_str=aux_str[:14]
            self.current_loss_str=aux_str
            self.states[0]=self.root_dir+'/'+self.name+"_"+self.current_loss_str
            current_model=self.states[0]
            print("Best checkpoint found for "+prev_model)
            save_model = [prev_model,current_model]
        self._update_registry(it,train_v,test_v,self.current_loss_str)
        return save_model

    def Best(self):
        return self.states[0]

    def _update_registry(self,it,train_v,test_v,min_test_v):
         with open(self.root_dir+'/'+self.name+".txt", 'a') as f:
            f.write(str(it)+" "+str(train_v)+" "+str(test_v)+" "+min_test_v+"\n")