### Run predictions

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

from GraphFromPDB import graph_from_pdb, prepare_batch, Dataset
from Model import MultiHeadGraphAttention, TransformerEncoderReadout, GraphAttentionNetwork

import tqdm
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)


# Data Preprocess
df = pd.read_csv("./example_data/data.csv")  #path to your CSV file
seq_length = [len(i) for i in df.seq.values]
U50_embeddings = np.load('./example_data/data_U50.npy')  #path to your NPY file
U50_embeddings_list = []
start_idx = 0
for length in seq_length:
    end_idx = start_idx + length
    protein_slice = U50_embeddings[start_idx:end_idx, :]
    U50_embeddings_list.append(protein_slice)
    start_idx = end_idx
    

# Dataset
x_pred = graph_from_pdb(df,U50_embeddings_list)
y_pred = df.label
pred_dataset = Dataset(x_pred, y_pred)


# Define hyper-parameters
HIDDEN_UNITS = 10
NUM_HEADS = 6
NUM_LAYERS = 1
BATCH_SIZE = 32


# Build model
gat_model = GraphAttentionNetwork(1024, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, BATCH_SIZE)


# Load Model
gat_model.load_weights('pgat_abpp.h5')


# Predict
predictions = gat_model.predict(
    pred_dataset
)
y = (predictions > 0.5).astype(int)
y =  np.ravel(y)
print('Prediction results:', y)