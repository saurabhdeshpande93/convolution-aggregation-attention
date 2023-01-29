import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def org_order(reordered_dof,dof,dim):

        '''
        For reordering the entries of array. Can be done efficiently by np.transpose options.

        Inputs:

        - reordered_dof : reordered array in the form (1x, 2x, ... , nx, 1y, 2y ...ny ,1z, ....,nz)
        - dof : degrees of freedom of the problem
        - dim : dimensionality

        Outputs:

        - Returns original ordering in the form  (1x, 1y, 1z, 2x, 2y, 2z ...,nx,ny,nz)

        '''

        original_dof = np.zeros((dof,))

        if dim==2:

          reordered_dof_x = reordered_dof[0:int(dof/dim)]
          reordered_dof_y = reordered_dof[int(dof/dim):dof]

          for i in range(int(dof/dim)):
             original_dof[2*i] = reordered_dof_x[i]
             original_dof[2*i+1] = reordered_dof_y[i]


        elif dim==3:

          reordered_dof_x = reordered_dof[0:int(dof/dim)]
          reordered_dof_y = reordered_dof[int(dof/dim):int(2*dof/dim)]
          reordered_dof_z = reordered_dof[int(2*dof/dim):dof]

          for i in range(int(dof/dim)):
             original_dof[3*i] = reordered_dof_x[i]
             original_dof[3*i+1] = reordered_dof_y[i]
             original_dof[3*i+2] = reordered_dof_z[i]

        return original_dof



def max_nodaldisps(labels_test,dof,dim):

    '''
    Input:

    - labels_test: labels of the test set
    - dof: degrees of freedom
    - dim: dimensionality

    Ouput:

    - max_nodaldisps: array of maximum nodal displacements of test examples

    '''

    n_test = len(labels_test)
    max_disps = np.zeros(n_test)

    for i in range(n_test):
        labels =  org_order(labels_test[i],dof,dim)
        lables =  labels.reshape(int(dof/dim),dim)
        norm = np.linalg.norm(lables,axis=1)
        max_disps[i] = np.max(abs(norm))

    return max_disps
