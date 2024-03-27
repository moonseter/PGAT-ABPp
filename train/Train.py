###Train the model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
permuted_indices = np.random.permutation(np.arange(df.shape[0]))

train_index = permuted_indices[: int(df.shape[0] * 0.8)]
U50_embeddings_train = [U50_embeddings_list[i] for i in train_index]
x_train = graph_from_pdb(df.iloc[train_index],U50_embeddings_train)
y_train = df.iloc[train_index].label

valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.9)]
U50_embeddings_valid = [U50_embeddings_list[i] for i in valid_index]
x_valid = graph_from_pdb(df.iloc[valid_index],U50_embeddings_valid)
y_valid = df.iloc[valid_index].label

test_index = permuted_indices[int(df.shape[0] * 0.9) :]
U50_embeddings_test = [U50_embeddings_list[i] for i in test_index]
x_test = graph_from_pdb(df.iloc[test_index],U50_embeddings_test)
y_test = df.iloc[test_index].label

train_dataset = Dataset(x_train, y_train)
valid_dataset = Dataset(x_valid, y_valid)
test_dataset = Dataset(x_test, y_test)


# Define hyper-parameters
HIDDEN_UNITS = 10
NUM_HEADS = 6
NUM_LAYERS = 1
BATCH_SIZE = 32
NUM_EPOCHS = 500
LEARNING_RATE = 1e-4

early_stopping = EarlyStopping(
    monitor="val_binary_accuracy", min_delta=1e-2, patience=50, restore_best_weights=True
)

savebestmodel = keras.callbacks.ModelCheckpoint('pgat_abpp.h5',save_best_only=True,monitor='val_binary_accuracy',verbose=1)  #path for saved model


# Build model
gat_model = GraphAttentionNetwork(1024, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, BATCH_SIZE)


# Compile model
gat_model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',  
                  metrics = ['binary_accuracy'])


# Fit
history = gat_model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping,savebestmodel],
    verbose=2,
)


# Evaluate
test_loss, test_accuracy = gat_model.evaluate(
    test_dataset,
    verbose=0
)

gat_model.summary()