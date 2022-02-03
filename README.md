# ECG_ResNet
- This repository contains the notebook code for classifying 12-Lead ECGs using Residual Neural Networks (ResNet). Some project files have been set as public, 
- Residual Neural Network development utilised the PyTorch library.
- For this project, the raw data was public, which contained 12-lead ECG data for thousands of patients from a variety of different sources. This data can be found on: https://moody-challenge.physionet.org/2020/

# Files
- training_data_experimentation - Initial experimentation with data to familiarise and understand the raw data
- creating_final_dataset.ipynb - Extracting ECG data from the header files to create a dataset for modelling
- resnet.py - Residual Neural Network implementation and testing in a python script file
- resnet.ipynb - Jupyter Notebook format of resnet.py to break each step down


# Future Considerations
- Make use of data augmentation techniques to address the major class imbalance problem in order to improve the performance of the ResNet. An option would be to augment the data via Generative Adversarial Networks (GANs).
