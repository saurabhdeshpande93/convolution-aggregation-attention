import Utils
import torch
import numpy as np
from datasets import load_dataset
import DatasetHandler
import os

import Model
import time


srcpath = os.path.dirname(os.getcwd())

case="2D"

if case == "3D":
	datapath = srcpath + "/frontiers_data/FEMData/"
	name = "3D"
	weights_path = srcpath+"/frontiers_data/saved_models/3dperceiver.pt"
	prediction_path = srcpath+"/frontiers_data/predictions/3dperceiver_predicts.npy"
	in_size=5835  #Degrees of freedom of the 3D problem
elif case == "2D":
	datapath = srcpath + "/frontiers_data/FEMData/"
	name = "2D"
	weights_path = srcpath+"/frontiers_data/saved_models/2dperceiver.pt"
	prediction_path = srcpath+"/frontiers_data/predictions/2dperceiver_predicts.npy"
	in_size=512  #Degrees of freedom of the 2D problem


perceiver = Model.SCCPerceiver(in_size)
perceiver.LoadDataset(datapath,name,trainable=True)
perceiver.Load(weights_path)
perceiver.SaveTestInference(prediction_path)
