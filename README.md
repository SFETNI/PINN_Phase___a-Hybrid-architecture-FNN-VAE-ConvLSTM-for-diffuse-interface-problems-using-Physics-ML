# PINN_Phase

![Repository Status](https://img.shields.io/badge/status-active-brightgreen)

## Abstract
**PINN_Phase** is a repository dedicated to the study and simulation of phase field modeling using Physics-Informed Neural Networks (PINNs). This approach leverages neural networks to solve partial differential equations related to phase transitions, making it a useful tool for applications in material science, such as grain growth simulation and solidification processes. The repository implements various machine learning models and architectures, including variational autoencoders (VAE) and convolutional LSTMs, for accurate phase field predictions. This work contributes to ongoing research in PINNs and neural approximations of complex physical processes.

## Associated Article (preprint)
This repository accompanies the research article titled **"[Pinn-Phase: A Physics-Informed Neural Network Hybrid Framework for Energy-Based Transfer Learning in Diffuse Interface Problems]"** by **[Seifallah Elfetni and Reza Darvishi Kamachali]**. Please refer to the article for a detailed theoretical background, methodology, and results. The code here supports the experiments and results discussed in the paper.

### Link to the article:
[Link](https://ssrn.com/abstract=5207041)  
or  
[DOI Link](http://dx.doi.org/10.2139/ssrn.5207041)

## Citation (if you find this study helpful):
@article{ELFETNI2025PinnPhase,  
  title = {Pinn-Phase: A Physics-Informed Neural Network Hybrid Framework for Energy-Based Transfer Learning in Diffuse Interface Problems},  
  author = {Seifallah Elfetni and Reza Darvishi Kamachali},  
  journal = {SSRN},  
  year = {2025},  
  url = {https://ssrn.com/abstract=5207041},  
  doi = {http://dx.doi.org/10.2139/ssrn.5207041},  
}

## Repository Structure

- **`main.py`**: This is the primary script for training and testing the models on the phase field simulation tasks. Detailed usage is provided below.
  
- **`L_PINN.py`**: Contains the model initialization, the main forward function, and the main training loop.

- **Modules Folder** (`Modules`): Contains individual Python scripts that serve as helper modules for the main program. These modules handle tasks like data preprocessing, network architecture definition, and utility functions.

- **Models Folder** (`models`): Houses pre-trained models and ground truth solutions required for benchmarking. Download any additional necessary files as outlined in the `Getting Started` section.

- **`VAE_ConvLSTM.ipynb`**: Jupyter notebook implementing VAE with ConvLSTM for further analysis and experimentation.

- **`figures/` and `VAE_figs/`**: These folders contain visualizations generated during experiments, including sample predictions and loss curves (real-time processing).

## PINN Phase Architecture

![PINN Phase Framework](https://github.com/SFETNI/PINN_Phase/blob/main/PINN_Phase.jpg)

## Getting Started

### Prerequisites
Ensure that you have Python 3.8+ and the necessary packages installed. Use the provided `requirements.txt` (if available) or manually install required packages.



