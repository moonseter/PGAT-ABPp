###Prepare dataset for model

import tensorflow as tf
import numpy as np

def graph_from_pdb(df,U50_embeddings_list):

    edges_list = []
    for A in df.A:
        edges = []
        A = np.array(eval(A))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] == 1:
                    edges.append([i,j])
        edges = np.array(edges)
        edges_list.append(edges)
    
    node_states_list = U50_embeddings_list   
    
    return tf.ragged.constant(node_states_list, dtype = tf.float32), tf.ragged.constant(edges_list, dtype = tf.int64)
 
 
def prepare_batch(x_batch, y_batch):

    (node_states, edges) = x_batch
    labels = y_batch
    
    num_nodes = node_states.row_lengths()
    num_bonds = edges.row_lengths()
    
    prot_indices = tf.range(len(num_nodes))
    prot_indicator = tf.repeat(prot_indices, num_nodes)
    
    gather_indices = tf.repeat(prot_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_nodes[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    edges = edges.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    edges = edges + increment[:, tf.newaxis]
    node_states = node_states.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    
    return (node_states, edges, prot_indicator), labels
  
  
def Dataset(X, y, batch_size = 32, shuffle = False):
    node_states_flat, edges_flat = X
    dataset = tf.data.Dataset.from_tensor_slices(((node_states_flat, edges_flat), y))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)