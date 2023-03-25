## Convolution, aggregation and attention based deep neural networks  

This repository provides the implementations of "Convolution, aggregation and attention based deep neural networks (DNNs) for accelerating simulations in mechanics". In this work, we propose different types of deep learning surrogate models for non-linear FEM simulations. For given input forces, DNNs predict deformed meshes as shown in the schematic below.

Proposed DNNs are trained on numerically generated non-linear FEM datasets (scripts are provided in the [<span style="color:blue">fem</span>](src/fem) directory. Datasets used in the paper are available on [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7585319.svg)](https://doi.org/10.5281/zenodo.7585319).

<br />

![schematic](schematic.jpg)

<br />

## Instructions

1. Download the supplementary data from zenodo and keep it in the src directory.

2. Neural network models can be used for inference using the pre-trained weights or can be trained from scratch by running `main.py` scripts in respective directories.

3. Use `postprcoess.py` to save the example of interest to further visualise it in Acegen using Mathematica notebooks present in the [<span style="color:blue">visualisation</span>](src/postprocess/visualisation) directory.

<br />

## Dependencies

Scripts have been tested running under Python 3.10.9, with the following packages installed (along with their dependencies). In addition, CUDA 10.1 and cuDNN 7 have been used.


- `tensorflow-gpu==2.4.1`
- `keras==2.4.3`
- `torch==1.13.1`
- `torchvision==0.10.0`
- `tqdm==4.64.1`
- `transformers==4.21.2`
- `numpy==1.24.2`
- `pandas==1.3.4`
- `scikit-learn==1.0.1`
- `matplotlib==3.5.0`

All the finite element simulations are performed using the [AceFEM](http://symech.fgg.uni-lj.si/Download.htm) library.

<br />

## Cite

Consider citing our paper if you use this code in your own work:

```
@misc{DESHPANDE221201386,
  doi = {10.48550/ARXIV.2212.01386},
  url = {https://arxiv.org/abs/2212.01386},
  author = {Deshpande, Saurabh and Sosa, Raúl I. and Bordas, Stéphane P. A. and Lengiewicz, Jakub},
  keywords = {Machine Learning (cs.LG), Computational Engineering, Finance, and Science (cs.CE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Convolution, aggregation and attention based deep neural networks for accelerating simulations in mechanics},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

<br />

## References

- Saurabh Deshpande, Jakub Lengiewicz and Stéphane P.A. Bordas. **MAgNET: A Graph U-Net Architecture for Mesh-Based Simulations**. *arXiv 2022*. https://arxiv.org/abs/2211.00713

- Saurabh Deshpande, Jakub Lengiewicz and Stéphane P.A. Bordas. **Probabilistic Deep Learning for Real-Time Large Deformation Simulations**. *Computer Methods in Applied Mechanics and Engineering (CMAME) 2022*. https://doi.org/10.1016/j.cma.2022.115307
