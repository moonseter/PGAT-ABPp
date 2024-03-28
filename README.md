# PGAT-ABPp
The implementation of the paper "PGAT-ABPp: Harnessing Protein Language Models and Graph Attention Networks for Antibacterial Peptide Identification with Remarkable Accuracy".

# Requirements
* Python 3.8.18
* Tensorflow 2.13.1
* Biopython 
* Pandas
* numpy
* matplotlib
* sklearn


# Usage
We implement our graph neural network models using the Tensorflow 2.13.1 deep learning framework, and training the models on Nvidia RTX4080 GPU.


# Training the model
1. To initiate the process with peptide sequences, PDB results can be obtained from the following links:  https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb or https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/batch/AlphaFold2_batch.ipynb.
2. To train the PGAT-ABPp model, the PDB files need to be prepared in CSV format using PDBProcess.py. Get the peptide embeddings acquired through ProtT5, which can be accessible via https://github.com/agemagician/ProtTrans/blob/master/Embedding/TensorFlow/Advanced/ProtT5-XL-UniRef50.ipynb. The example data files are provided named data.csv and data_U50.npy in the example_data folder.
3. Train.py is provided for an example usage.


# Using trained model for ABPs prediction
1. Applying the PGAT-ABPp to a new peptide requires the preparation of data files in CSV format and NPY format. To understand the process, data.csv and data_U50.npy are provided in the example_data folder.
2. The well trained model pig_abppred.h5 is supplied in the predict folder. Predict.py is provided for an example usage.


# Notification of commercial use
Commercialization of this product is prohibited without our permission.