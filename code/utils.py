import pickle
import numpy as np
# import prepFeatures as pf
import aaindex
from Bio import SeqIO
from Bio import AlignIO
from Bio.Align.Applications import MafftCommandline
from multiprocessing import Pool

def mix_subtypes_seqs():
    """
    Combine sequences of datasets H1N1, H3N2, H5N1 and H9N2 into one.
    """
    seqs_H1N1 = list(SeqIO.parse("../data/H1N1_seqs", "fasta"))
    for s in seqs_H1N1:
        s.id = s.id + "/H1N1"
    
    seqs_H3N2 = list(SeqIO.parse("../data/H3N2_seqs", "fasta"))
    for s in seqs_H3N2:
        s.id = s.id + "/H3N2"
    
    seqs_H5N1 = list(SeqIO.parse("../data/H5N1_seqs", "fasta"))
    for s in seqs_H5N1:
        s.id = s.id + "/H5N1"
    
    seqs_H9N2 = list(SeqIO.parse("../data/H9N2_seqs", "fasta"))
    for s in seqs_H9N2: 
        s.id = s.id + "/H9N2"
    
    mixed_seqs = seqs_H1N1 + seqs_H3N2 + seqs_H5N1 + seqs_H9N2
    
    with open("../data/mixed_seqs.fas", "w") as f:
        SeqIO.write(mixed_seqs, f, "fasta" )
    
    return seqs_H1N1, seqs_H3N2, seqs_H5N1, seqs_H9N2, mixed_seqs

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def mix_subtypes_dist():
    """
    Combine antigenic distance pairs of datasets H1N1, H3N2, H5N1 and H9N2 into one.
    """
    fdist_H1N1 = "../data/H1N1dist_pickle"
    with open(fdist_H1N1, "rb") as f:
        dh1n1 = pickle.load(f)
    for k in dh1n1.keys():
        vnames = k.split(",")
        v1 = vnames[0]+"/H1N1"
        v2 = vnames[1]+"/H1N1"
        vnames = v1 + "," + v2
        dh1n1[vnames] = dh1n1.pop(k)
    # print dh1n1.keys()
    # a = len(dh1n1.keys())

    fdist_H3N2 = "../data/H3N2dist_pickle"
    with open(fdist_H3N2, "rb") as f:
        dh3n2 = pickle.load(f)
    for k in dh3n2.keys():
        vnames = k.split(",")
        v1 = vnames[0]+"/H3N2"
        v2 = vnames[1]+"/H3N2"
        vnames = v1 + "," + v2
        dh3n2[vnames] = dh3n2.pop(k)
    # print dh3n2.keys()
    # b = len(dh3n2.keys())

    fdist_H5N1 = "../data/H5N1dist_pickle"
    with open(fdist_H5N1, "rb") as f:
        dh5n1 = pickle.load(f)
    for k in dh5n1.keys():
        vnames = k.split(",")
        v1 = vnames[0] + "/H5N1"
        v2 = vnames[1] + "/H5N1"
        vnames = v1 + "," + v2
        dh5n1[vnames] = dh5n1.pop(k)
    # print dh5n1.keys()
    # c = len(dh5n1.keys())

    fdist_H9N2 = "../data/H9N2dist_pickle"
    with open(fdist_H9N2, "rb") as f:
        dh9n2 = pickle.load(f)
    for k in dh9n2.keys():
        vnames = k.split(",")
        v1 = vnames[0] + "/H9N2"
        v2 = vnames[1] + "/H9N2"
        vnames = v1 + "," + v2
        dh9n2[vnames] = dh9n2.pop(k)
    # print dh9n2.keys()
    # d = len(dh9n2.keys())
        
    d1 = merge_two_dicts(dh1n1, dh3n2)
    d2 = merge_two_dicts(dh5n1, dh9n2)
    dist_dict = merge_two_dicts(d1, d2)
    
    return dh1n1, dh3n2, dh5n1, dh9n2, dist_dict

def load_aligned_seqs():
    # seqs_H1N1, seqs_H3N2, seqs_H5N1, seqs_H9N2, _ = mix_subtypes_seqs()
    # num_H1N1 = len(seqs_H1N1)
    # num_H3N2 = len(seqs_H3N2)
    # num_H5N1 = len(seqs_H5N1)
    # num_H9N2 = len(seqs_H9N2)

    num_H1N1 = 68
    num_H3N2 = 621
    num_H5N1 = 148
    num_H9N2 = 29
    
    mixed_seqs = list(SeqIO.parse("../data/align_mixed_seqs.fas", "fasta"))

    assert (num_H1N1+num_H3N2+num_H5N1+num_H9N2) == len(mixed_seqs)

    start = 0
    end = num_H1N1
    seqs_H1N1 = mixed_seqs[start:end]
    
    start = end
    end = end+num_H3N2
    seqs_H3N2 = mixed_seqs[start:end]
    
    start = end
    end = end+num_H5N1
    seqs_H5N1 = mixed_seqs[start:end]
    
    start = end
    end = end+num_H9N2
    seqs_H9N2 = mixed_seqs[start:end]
    
    return seqs_H1N1, seqs_H3N2, seqs_H5N1, seqs_H9N2, mixed_seqs

## Test load_aligned_seqs(): 
# seqs_H1N1, seqs_H3N2, seqs_H5N1, seqs_H9N2, mixed_seqs = load_aligned_seqs()
# print seqs_H1N1[0]
# print seqs_H3N2[0]
# print seqs_H5N1[0]
# print seqs_H9N2[0]


def remove_inconsistency(seqs, dist, pf_dfile):
    XNamelist = dist.keys()
    YList = dist.values()
    
    with open(pf_dfile, "rb") as f:
        pf_dict = pickle.load(f)
    pfNamelist = pf_dict.keys()
    
    X = []
    Y = []
    XName = []
    
    for i in range(len(XNamelist)):
        vname = XNamelist[i].split(",")
        alias = vname[1] + "," + vname[0]
        a1 = XNamelist[i] in pfNamelist
        a2 = alias in pfNamelist
        if (not(a1 or a2)):
            print(str(vname) + " is not in the list of pairwise features")
        
        if a1:
            X.append(pf_dict[XNamelist[i]])
            XName.append(XNamelist[i])
            Y.append(YList[i])
        if a2:
            X.append(pf_dict[alias])
            XName.append(alias)
            Y.append(YList[i])
    assert len(X) == len(XName)
    assert len(X) == len(Y)
    
    return X, Y, XName


def prep_dataset(indexID):
    """
    
    """
    seqs_H1N1, seqs_H3N2, seqs_H5N1, seqs_H9N2, mixed_seqs = load_aligned_seqs()
    dh1n1, dh3n2, dh5n1, dh9n2, dist_dict = mix_subtypes_dist()
    
    pf_dfile = "../data/pairFeatures_" + indexID

    X_h1n1, Y_h1n1, XName_h1n1 = remove_inconsistency(seqs_H1N1, dh1n1, pf_dfile)
    X_h3n2, Y_h3n2, XName_h3n2 = remove_inconsistency(seqs_H3N2, dh3n2, pf_dfile)
    X_h5n1, Y_h5n1, XName_h5n1 = remove_inconsistency(seqs_H5N1, dh5n1, pf_dfile)
    X_h9n2, Y_h9n2, XName_h9n2 = remove_inconsistency(seqs_H9N2, dh9n2, pf_dfile)
    X = X_h1n1 + X_h3n2 + X_h5n1 + X_h9n2
    Y = Y_h1n1 + Y_h3n2 + Y_h5n1 + Y_h9n2
    XName = XName_h1n1 + XName_h3n2 + XName_h5n1 + XName_h9n2
    
    
    dataset_path= "../data/dataset_pickle/"
    # Write X, Y and XName into files ---- H1N1
    X_h1n1_pickle = dataset_path + indexID + "_h1n1_X.pickle"
    Y_h1n1_pickle = dataset_path + indexID + "_h1n1_Y.pickle"
    XName_h1n1_pickle = dataset_path + indexID + "_h1n1_XName.pickle"

    with open(X_h1n1_pickle, "wb") as f:
        pickle.dump(X_h1n1, f)
    with open(Y_h1n1_pickle, "wb") as f:
        pickle.dump(Y_h1n1, f)
    with open(XName_h1n1_pickle, "wb") as f:
        pickle.dump(XName_h1n1, f)
        
    # H3N2
    X_h3n2_pickle = dataset_path + indexID + "_h3n2_X.pickle"
    Y_h3n2_pickle = dataset_path + indexID + "_h3n2_Y.pickle"
    XName_h3n2_pickle = dataset_path + indexID + "_h3n2_XName.pickle"

    with open(X_h3n2_pickle, "wb") as f:
        pickle.dump(X_h3n2, f)
    with open(Y_h3n2_pickle, "wb") as f:
        pickle.dump(Y_h3n2, f)
    with open(XName_h3n2_pickle, "wb") as f:
        pickle.dump(XName_h3n2, f)
    
    # H5N1
    X_h5n1_pickle = dataset_path + indexID + "_h5n1_X.pickle"
    Y_h5n1_pickle = dataset_path + indexID + "_h5n1_Y.pickle"
    XName_h5n1_pickle = dataset_path + indexID + "_h5n1_XName.pickle"

    with open(X_h5n1_pickle, "wb") as f:
        pickle.dump(X_h5n1, f)
    with open(Y_h5n1_pickle, "wb") as f:
        pickle.dump(Y_h5n1, f)
    with open(XName_h5n1_pickle, "wb") as f:
        pickle.dump(XName_h5n1, f)
        
    # H9N2
    X_h9n2_pickle = dataset_path + indexID + "_h9n2_X.pickle"
    Y_h9n2_pickle = dataset_path + indexID + "_h9n2_Y.pickle"
    XName_h9n2_pickle = dataset_path + indexID + "_h9n2_XName.pickle"

    with open(X_h9n2_pickle, "wb") as f:
        pickle.dump(X_h9n2, f)
    with open(Y_h9n2_pickle, "wb") as f:
        pickle.dump(Y_h9n2, f)
    with open(XName_h9n2_pickle, "wb") as f:
        pickle.dump(XName_h9n2, f)
        
    # Mixed
    X_pickle = dataset_path + indexID + "_X.pickle"
    Y_pickle = dataset_path + indexID + "_Y.pickle"
    XName_pickle = dataset_path + indexID + "_XName.pickle"

    with open(X_pickle, "wb") as f:
        pickle.dump(X, f)
    with open(Y_pickle, "wb") as f:
        pickle.dump(Y, f)
    with open(XName_pickle, "wb") as f:
        pickle.dump(XName, f)
    
    print("Dataset has: " + str(len(X)) + " examples. \n")


def load_dataset(indexID):
    dataset_path = "../data/dataset_pickle/"
    
    X_h1n1_pickle = dataset_path + indexID + "_h1n1_X.pickle"
    Y_h1n1_pickle = dataset_path + indexID + "_h1n1_Y.pickle"
    XName_h1n1_pickle = dataset_path + indexID + "_h1n1_XName.pickle"
    with open(X_h1n1_pickle, "rb") as f:
        X_h1n1 = pickle.load(f)
        X_h1n1 = np.array(X_h1n1)
    with open(Y_h1n1_pickle, "rb") as f:
        Y_h1n1 = pickle.load(f)
        Y_h1n1 = np.array(Y_h1n1).reshape((len(Y_h1n1), 1))   
    with open(XName_h1n1_pickle, "rb") as f:
        XName_h1n1 = pickle.load(f)
        XName_h1n1 = np.array(XName_h1n1).reshape((len(XName_h1n1), 1))
    
    
    X_h3n2_pickle = dataset_path + indexID + "_h3n2_X.pickle"
    Y_h3n2_pickle = dataset_path + indexID + "_h3n2_Y.pickle"
    XName_h3n2_pickle = dataset_path + indexID + "_h3n2_XName.pickle"
    with open(X_h3n2_pickle, "rb") as f:
        X_h3n2 = pickle.load(f)
        X_h3n2 = np.array(X_h3n2)       
    with open(Y_h3n2_pickle, "rb") as f:
        Y_h3n2 = pickle.load(f)
        Y_h3n2 = np.array(Y_h3n2).reshape((len(Y_h3n2), 1))      
    with open(XName_h3n2_pickle, "rb") as f:
        XName_h3n2 = pickle.load(f)
        XName_h3n2 = np.array(XName_h3n2).reshape((len(XName_h3n2), 1))
        
        
    X_h5n1_pickle = dataset_path + indexID + "_h5n1_X.pickle"
    Y_h5n1_pickle = dataset_path + indexID + "_h5n1_Y.pickle"
    XName_h5n1_pickle = dataset_path + indexID + "_h5n1_XName.pickle"
    with open(X_h5n1_pickle, "rb") as f:
        X_h5n1 = pickle.load(f)
        X_h5n1 = np.array(X_h5n1)
    with open(Y_h5n1_pickle, "rb") as f:
        Y_h5n1 = pickle.load(f)
        Y_h5n1 = np.array(Y_h5n1).reshape((len(Y_h5n1), 1))
    with open(XName_h5n1_pickle, "rb") as f:
        XName_h5n1 = pickle.load(f)
        XName_h5n1 = np.array(XName_h5n1).reshape((len(XName_h5n1), 1))
        
    
    X_h9n2_pickle = dataset_path + indexID + "_h9n2_X.pickle"
    Y_h9n2_pickle = dataset_path + indexID + "_h9n2_Y.pickle"
    XName_h9n2_pickle = dataset_path + indexID + "_h9n2_XName.pickle"
    with open(X_h9n2_pickle, "rb") as f:
        X_h9n2 = pickle.load(f)
        X_h9n2 = np.array(X_h9n2)       
    with open(Y_h9n2_pickle, "rb") as f:
        Y_h9n2 = pickle.load(f)
        Y_h9n2 = np.array(Y_h9n2).reshape((len(Y_h9n2), 1))      
    with open(XName_h9n2_pickle, "rb") as f:
        XName_h9n2 = pickle.load(f)
        XName_h9n2 = np.array(XName_h9n2).reshape((len(XName_h9n2), 1))
        
    
    X_pickle = dataset_path + indexID + "_X.pickle"
    Y_pickle = dataset_path + indexID + "_Y.pickle"
    XName_pickle = dataset_path + indexID + "_XName.pickle"
    with open(X_pickle, "rb") as f:
        X = pickle.load(f)
        X = np.array(X)       
    with open(Y_pickle, "rb") as f:
        Y = pickle.load(f)
        Y = np.array(Y).reshape((len(Y), 1))      
    with open(XName_pickle, "rb") as f:
        XName = pickle.load(f)
        XName = np.array(XName).reshape((len(XName), 1))
    
    dataset = {}
    dataset["h1n1"] = (X_h1n1, Y_h1n1, XName_h1n1)
    dataset["h3n2"] = (X_h3n2, Y_h3n2, XName_h3n2)
    dataset["h5n1"] = (X_h5n1, Y_h5n1, XName_h5n1)
    dataset["h9n2"] = (X_h9n2, Y_h9n2, XName_h9n2)
    dataset["mixed"] = (X, Y, XName)
    
    return dataset


def create_binary_labels(Y, threshold = 4):
    """
    Create binary labels of influenza strain pairs. 
    args: antigenic distances of influenza pairs Y
    threshould:
        if dist < threshold, the strain pairs are labeled as "similar" (0)
        if dist >= threshold, the strain pairs are labeled as "distinct" (1)
        the default threshold is 4 (ref: Liao2008Bioinformatics)
    Return a binary list of Y indicating the similarity of influenza strain pairs.
    """
    labels = []
    for i in Y:
        if i < 4:
            labels.append(0)
        else:
            labels.append(1)
    labels = np.array(labels).flatten()
    # print "type(labels): " + str(type(labels))
    
    return labels


def shuffle_dataset_with_labels(X, Y, labels, XName):
    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X = X[s]
    Y = Y[s]
    labels= labels[s]
    XName = XName[s]
    
    return X, Y, labels, XName


def avoid_none(X, Y, penalty):
    """
    Ensure that X, Y do not contain None (raised from gaps).
    args: X, Y, penalty
    return: X, Y. None has been replaced with penalty
    """
    Y = Y.astype(np.float)
    
    if np.any(X == None):
        # print "*** Unattended gaps in this AAIndex matrix ***"
        # print " "
        X[X == None] = penalty
    assert not(np.any(X == None))
    assert not(np.any(Y == None))
    
    return X, Y


def split_train_test_with_labels(X, Y, labels, XName, train_portion):
    assert train_portion > 0
    assert train_portion < 1
    
    X = X.reshape(X.shape[0], -1)
    [m, n] = X.shape
    m_train = int(m*train_portion)
    m_test = m - m_train
    
    X_train = X[0: m_train, :]
    Y_train = Y[0: m_train]
    labels_train = labels[0: m_train]
    XName_train = XName[0: m_train]
    
    X_test = X[m_train:m, :]
    Y_test = Y[m_train:m]
    labels_test = labels[m_train: m]
    XName_test = XName[m_train:m]
    
    # print "# Train data: " + str(m_train)
    # print "# Test data: " + str(m_test)
    
    return X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test
