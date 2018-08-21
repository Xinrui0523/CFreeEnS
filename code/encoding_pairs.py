import pandas as pd
import numpy as np
import pickle
import aaindex
from Bio import SeqIO

aaindex.init("./aaindex/")

def comp_sites(a, b, idx):
    """
    Calculate dissimilarity between amino acids from AAIndex
    """
    x = aaindex.get(idx)
    m = x.get(a,a)
    n = x.get(b,b)
    k = x.get(a,b)
    
    NoneType = type(None)
    if isinstance(m, NoneType) or isinstance(n, NoneType) or isinstance(k, NoneType):
        # print "Unattended gaps in this AAIndex matrice"
        dist = None
    else:
        dist = m + n -2*k
    return dist

def pair_idx_encode(s1, s2, idx):
    """
    Calculate dissimilarity between two sequences
    """
    dist = []
    assert len(s1)==len(s2)
    for i in range(len(s1)):
        # print i
        tmp = comp_sites(s1[i], s2[i], idx)
        dist.append(tmp)

    return dist

def batch_idx_encode(seq_file, idx, out_file):

    seqList = list(SeqIO.parse(seq_file, "fasta"))
    n = len(seqList)
    
    feature_dict = {}
    
    for i in range(n):
        id1 = seqList[i].id
        seq1 = seqList[i].seq
        for j in range(i+1, n):
            id2 = seqList[j].id
            seq2 = seqList[j].seq

            feature_dict[id1+","+id2] = pair_idx_encode(seq1, seq2, idx)

    with open(out_file, "wb") as f:
        pickle.dump(feature_dict, f)

    return feature_dict

def batch_idxList_encode(seq_file, idxList):
    for idx in idxList:
        # print idx

        out_file = seq_file + "_" + idx
        batch_idx_encode(seq_file, idx, out_file)

    return True

