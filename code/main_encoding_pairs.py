from encoding_pairs import batch_idxList_encode

import pickle
import aaindex

aaindex.init('./aaindex/')

def encode_subtype(subtype):
    seq_file = '../data/' + subtype + '_seqs'
    with open('./aaindex/indexKeyList_2_pickle', 'rb') as f:
        idxList = pickle.load(f)
        
    batch_idxList_encode(seq_file, idxList)
    