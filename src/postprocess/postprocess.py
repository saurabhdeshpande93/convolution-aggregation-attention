import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from post_utils import org_order, max_nodaldisps

## 2d, 3d
case = '3d'

## cnn, magnet, perceiver
type = 'magnet'

## Load test data feature/labels
srcpath = os.path.dirname(os.getcwd())
datapath = srcpath + "/frontiers_data/FEMData/"

## Assign dimensionality and degrees of freedom of the problem.
if case=='2d':
   dim, dof = 2, 512
   X_test = np.load(datapath+'features_test_2D.npy')
   Y_test = np.load(datapath+'labels_test_2D.npy')

else:
   dim, dof = 3, 5835
   X_test = np.load(datapath+'features_test_3D.npy')
   Y_test = np.load(datapath+'labels_test_3D.npy')

## Load neural network predictions
pred_path = srcpath + "/frontiers_data/predictions/"

if type =='cnn':
    predictions = np.load(str(pred_path)+str(case)+'cnn_predicts.npy')
elif type == 'perceiver':
    predictions = np.load(str(pred_path)+str(case)+'perceiver_predicts.npy')
else:
    predictions = np.load(str(pred_path)+str(case)+'magnet_predicts.npy')


## Error metrics for the test set
error = np.abs(predictions - Y_test)

print("Average error of set is ", np.mean(error))
print("Max error of set is ", np.max(error))


## Find out the example with maximum nodal displacement
max_disps = max_nodaldisps(Y_test,dof,dim)
index= np.argmax(max_disps)


## Export a particular test example for visualisation in acegen
data = np.zeros((dof,4))
data[:,0] = org_order(X_test[index],dof,dim)
data[:,1] = org_order(predictions[index],dof,dim)
data[:,2] = org_order(Y_test[index],dof,dim)
data[:,3] = org_order(error[index],dof,dim)

np.savetxt('visualisation/'+str(case)+str(type)+'_t'+str(index)+'.csv', data, delimiter=",")
