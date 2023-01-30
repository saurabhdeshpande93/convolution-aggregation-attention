## Non-linear FEM data generation

The datasets used in this paper are already provided in the frontiers_data folder. Here we are providing FEM data generation scripts if the user wishes to generate their own.

First, we generate force-displacement datasets by randomly applying point/body forces on the given mesh and the entire data is saved in a single .csv file. And then we preprocess it to convert data into training/test arrays to be fed to the network for training.

Non-linear FEM datasets are generated using the AceFEM library. AceFEM is a general finite element system for Mathematica that effectively combines symbolic and numeric approaches. It can be downloaded from [here](http://symech.fgg.uni-lj.si/Download.htm).
