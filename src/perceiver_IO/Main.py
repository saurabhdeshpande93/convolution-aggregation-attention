import Utils
import torch
import numpy as np
from datasets import load_dataset
import DatasetHandler


import Model
import time

print(torch.cuda.is_available())

np.set_printoptions(threshold=np.inf)


name = "Data/elephant"

perceiver = Model.SCCPerceiver()
perceiver.LoadDataset(name,trainable=True)

# perceiver.TestLoss()
#perceiver.Train(500000,1000)
perceiver.Load("SavedModels/elephant/elephant_bz64_0.002614192564")
# perceiver.Train(500000,1000)
startTime = time.time()
perceiver.Train(500000,1000)
# perceiver.TestInference()
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
# print(perceiver.Eval(custom=True))
