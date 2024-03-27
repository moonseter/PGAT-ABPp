###Process PDB files to obtain CSV format file

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pandas as pd
import os

def load_pdb(pdbfile):

    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('.')[0],pdbfile)
    residues = [r for r in structure.get_residues()]
    
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]
    
    distances = np.empty((len(residues),len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one-two)
            
    return distances, seqs[0]


def load_cmap(pdbfile, cmap_thresh):

    D, seq = load_pdb(pdbfile)
    
    A = np.zeros(D.shape, dtype = np.float32)
    
    A[D < cmap_thresh] = 1
    
    return A, seq
    
    
def read_files(folder_path, label):

    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        A, seq = load_cmap(file_path, CMAP_THRESH)

        A_str = str(A.tolist())

        pdb_entry = {
                'filename': filename,
                'A': A_str,
                'seq': seq,
                'label': None
        }
        data.append(pdb_entry)
    return data


CMAP_THRESH = 10  #define cmap_thresh
folder = './example_data/files'  #path to PDB files
data = read_files(folder, None)
df = pd.DataFrame(data)

df.to_csv('./example_data/data.csv', index=False)  #path to your prepared data_to_load folder