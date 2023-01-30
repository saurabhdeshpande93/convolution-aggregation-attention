import Utils
import torch
import numpy as np
from datasets import load_dataset
import DatasetHandler
import os

import Model
import time

print(torch.cuda.is_available())

np.set_printoptions(threshold=np.inf)

srcpath = os.path.dirname(os.getcwd())
datapath = srcpath + "/frontiers_data/FEMData/"

prediction_path = srcpath+"/frontiers_data/predictions/"
weights_path = srcpath+"/saved_models/"

name = "Data/elephant"

perceiver = Model.SCCPerceiver()
perceiver.LoadDataset(name,trainable=True)

# perceiver.TestLoss()
#perceiver.Train(500000,1000)
perceiver.Load("SavedModels/elephant/elephant_bz64_0.002614192564")

#
perceiver.Train(500000,1000)
