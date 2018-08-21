from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import mix_subtypes_dist, create_binary_labels
from utils import shuffle_dataset_with_labels, split_train_test_with_labels

import numpy as np
import pickle
import aaindex

from Bio import SeqIO
from heapq import nlargest
from itertools import compress
from sklearn.naive_bayes import GaussianNB

E1 = [129,130,131,155,156,157,158,159,160,196]
E2 = [126,127,128,132,133,134,135,136,153,162,163,165,188,189,190,192,193,194,197,198,199,201,248]
E3 = [74,100,101,122,123,124,125,137,138,140,141,142,143,144,145,147,149,150,166,167,168,186,187,211,212,213,214,215,216,217,219,222,223,224,225,226,227,233,234,244,255,257]
E4 = [63,65,75,77,78,79,80,81,82,93,94,95,96,102,103,104,105,106,109,119,120,121,169,171,172,174,175,207,208,209,210,236,238,239,240,242,259,260]
E5 = [57,58,59,60,62,83,85,89,90,91,92,110,114,173,261,262,263,264,267,269,271]
E6 = [49,50,53,54,55,56,272,273,274,275,276,277,278,279,280,284,285,298,299,300,301]
E7 = [44,45,46,47,48,289,290,291,292,293,296,297,307,308,310,311,312]
E8 = [40,41,313,315]
E9 = [22,23,24,25,27,29,31,32,33,34,35,37,38,39,318]
E10 = [1,2,3,4,5,6,7,8,9,10,12,14,18,20,21,321,323,324,325,326,327,328]

featureMatrix_path = "../data/srep2017Peng_featMatrix/"


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
    
    mixed_seqs = list(SeqIO.parse("../data/align_mixed_seqs_numH3.fas", "fasta"))

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


def get_mutList(s1, s2):
	assert len(s1) == len(s2)
	mutsList = list(i[0] != i[1] for i in zip(s1, s2))
	# mutsList = np.array(mutations).astype(int)

	return mutsList

def count_muts(mutsList, regBand):

	mutIndexes = list(compress(xrange(len(mutsList)), mutsList))
	# print mutIndexes
	count = 0
	for index in mutIndexes:
		pos = index+1
		if pos in regBand:
			count += 1

	return count

def get_features(s1, s2):
	mutsList = get_mutList(s1, s2)
	regBands = [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10]

	counts = []

	for regBand in regBands:
		count = count_muts(mutsList, regBand)
		counts.append(count)

	return counts 

# AAIndex for hydrophobicity, volume, charge, polarity, and accessible surface area
hydro_index = "FASG890101" 
vol_index = "GRAR740103"
charge_index = "ZIMJ680104"
polarity_index = "CHAM820101"
asa_index = "JANJ780101"

physicochem_index_list = [hydro_index, vol_index, charge_index, polarity_index, asa_index]

physico_chem_theta_dict = {}
physico_chem_theta_dict["FASG890101"] = 1.82
physico_chem_theta_dict["GRAR740103"] = 54.67
physico_chem_theta_dict["ZIMJ680104"] = 2.49
physico_chem_theta_dict["CHAM820101"] = 34.87
physico_chem_theta_dict["JANJ780101"] = 0.10

def get_score(a, indexID):
	x = aaindex.get(indexID)
	if a == "-":
		score = 0
	else:
		score = x.get(a)
	return score


def get_physicochem_feature(s1, s2, indexID):
	"""
	hydro_index = "FASG890101" --> 1.82
	vol_index = "GRAR740103" --> 54.67
	charge_index = "ZIMJ680104" --> 2.49
	polarity_index = "CHAM820101" --> 34.87
	asa_index = "JANJ780101" --> 0.10
	"""

	theta = physico_chem_theta_dict[indexID]

	mutsList = get_mutList(s1, s2)
	mutIndexes = list(compress(xrange(len(mutsList)), mutsList))

	avg_change = 0
	chang_list = []
	# print len(mutIndexes)

	flag = False

	if len(mutIndexes) == 0:
		pass
	elif len(mutIndexes) <= 3:
		for i in mutIndexes:
			if s1[i] == "X" or s2[i] == "X":
				change = 0
			else:
				change = abs(get_score(s1[i], indexID) - get_score(s2[i], indexID))
			avg_change += change
		avg_change = avg_change/len(mutIndexes)
	else:
		for i in mutIndexes:
			if s1[i] == "X" or s2[i] == "X":
				change = 0
			else:
				change = abs(get_score(s1[i], indexID) - get_score(s2[i], indexID))
			chang_list.append(change)
			avg_change = np.array(nlargest(3, chang_list)).mean()

	if avg_change >= theta:
		flag = True
	
	return flag

def get_physicochem_features(s1, s2):
	physicochem_list = []
	for indexID in physicochem_index_list:
		flag = get_physicochem_feature(s1, s2, indexID)
		physicochem_list.append(flag)

	return physicochem_list

def seqList2Dict(seqs_list):
	"""
	"""
	seqs_dict = {}
	for s in seqs_list:
		vname = s.id
		vseq = s.seq
		seqs_dict[vname] = vseq
	return seqs_dict

def get_feature_matrix(seqs_list, dist_dict, outfile):
	"""
	"""
	XNamelist = dist_dict.keys()
	YList = dist_dict.values()

	seqs_dict = seqList2Dict(seqs_list)

	featureMatrix_dict = {}
	for name in XNamelist:
		vnames = name.split(",")
		v1 = vnames[0]
		v2 = vnames[1]
		s1 = seqs_dict[v1]
		s2 = seqs_dict[v2]

		regband_counts = get_features(s1, s2)
		physicochem_list = get_physicochem_features(s1, s2)

		feature_list = regband_counts + physicochem_list

		featureMatrix_dict[v1 + "," + v2] = feature_list

	with open(outfile, "w") as f:
		pickle.dump(featureMatrix_dict, f)

	return featureMatrix_dict

def write_feature_matrix():
	seqs_h1n1, seqs_h3n2, seqs_h5n1, seqs_h9n2, mixed_seqs = load_aligned_seqs()
	dh1n1, dh3n2, dh5n1, dh9n2, dmixed = mix_subtypes_dist()

	h1n1_mutMatrix = get_feature_matrix(seqs_h1n1, dh1n1, featureMatrix_path + "h1n1.feats")
	h3n2_mutMatrix = get_feature_matrix(seqs_h3n2, dh3n2, featureMatrix_path + "h3n2.feats")
	h5n1_mutMatrix = get_feature_matrix(seqs_h5n1, dh5n1, featureMatrix_path + "h5n1.feats")
	h9n2_mutMatrix = get_feature_matrix(seqs_h9n2, dh9n2, featureMatrix_path + "h9n2.feats")
	mixed_mutMatrix = get_feature_matrix(mixed_seqs, dmixed, featureMatrix_path + "mixed.feats")

def load_features_dataset(featureMatrix_file, dist_dict):
	with open(featureMatrix_file, "r") as f:
		featureMarix = pickle.load(f)

	X = []
	Y = []
	XName = []

	for vname in dist_dict.keys():
		X.append(featureMarix[vname])
		Y.append(dist_dict[vname])
		Y = [float(i) for i in Y]
		XName.append(vname)

	return X, Y, XName

def naiveBayes(outfile, Xtrain, Ytrain, labelsTrain, XNameTrain, Xtest, Ytest, labelsTest, XNameTest):
	"""
	"""

	model = GaussianNB().fit(Xtrain, labelsTrain)
	labels_train_est = model.predict(Xtrain)

	labels_test_est = model.predict(Xtest)

	train_accu = accuracy_score(labelsTrain, labels_train_est)
	train_prec = precision_score(labelsTrain, labels_train_est)
	train_rec = recall_score(labelsTrain, labels_train_est)
	train_f1 = f1_score(labelsTrain, labels_train_est)

	test_accu = accuracy_score(labelsTest, labels_test_est)
	test_prec = precision_score(labelsTest, labels_test_est)
	test_rec = recall_score(labelsTest, labels_test_est)
	test_f1 = f1_score(labelsTest, labels_test_est)

	result = {}
	result["model"] = model

	result["train_accu"] = train_accu
	result["train_prec"] = train_prec
	result["train_rec"] = train_rec
	result["train_f1"] = train_f1

	result["test_accu"] = test_accu
	result["test_prec"] = test_prec
	result["test_rec"] = test_rec
	result["test_f1"] = test_f1

	with open(outfile, "w") as f:
		pickle.dump(result, f)

	return result

def kfoldNB(k, prefix, X, Y, XName):
	for i in range(k):
		outfile = "../result/srep2017Peng/" + prefix + "_" + str(i) + ".nb"

		labels = create_binary_labels(Y, 4)
		
		X = np.array(X)
		Y = np.array(Y).reshape(len(Y),1)
		XName = np.array(XName).reshape(len(XName),1)
		labels = np.array(labels)

		X, Y, labels, XName = shuffle_dataset_with_labels(X, Y, labels, XName)
		Xtrain, Ytrain, labelsTrain, XNameTrain, Xtest, Ytest, labelsTest, XNameTest = split_train_test_with_labels(X, Y, labels, XName, 0.8)
		result = naiveBayes(outfile, Xtrain, Ytrain, labelsTrain, XNameTrain, Xtest, Ytest, labelsTest, XNameTest)

def trainNB(prefix, dist_dict):
	featureMatrix_file = featureMatrix_path + prefix + ".feats"
	X, Y, XName = load_features_dataset(featureMatrix_file, dist_dict)

	kfoldNB(10, prefix, X, Y, XName)

def transfer_learning_2():

	# Train on H1N1+H3N2; Test on H5N1+H9N2

	dh1n1, dh3n2, dh5n1, dh9n2, dmixed = mix_subtypes_dist()

	h1n1_fm_file = featureMatrix_path + "h1n1.feats"
	h3n2_fm_file = featureMatrix_path + "h3n2.feats"
	h5n1_fm_file = featureMatrix_path + "h5n1.feats"
	h9n2_fm_file = featureMatrix_path + "h9n2.feats"
	
	X_h1n1, Y_h1n1, XName_h1n1 = load_features_dataset(h1n1_fm_file, dh1n1)
	X_h3n2, Y_h3n2, XName_h3n2 = load_features_dataset(h3n2_fm_file, dh3n2)
	X_h5n1, Y_h5n1, XName_h5n1 = load_features_dataset(h5n1_fm_file, dh5n1)
	X_h9n2, Y_h9n2, XName_h9n2 = load_features_dataset(h9n2_fm_file, dh9n2)

	dataset = {}
	dataset["h1n1"] = (X_h1n1, Y_h1n1, XName_h1n1)
	dataset["h3n2"] = (X_h3n2, Y_h3n2, XName_h3n2)
	dataset["h5n1"] = (X_h5n1, Y_h5n1, XName_h5n1)
	dataset["h9n2"] = (X_h9n2, Y_h9n2, XName_h9n2)

	Xtrain2 = np.concatenate((X_h1n1, X_h3n2), axis = 0)
	Ytrain2 = np.concatenate((Y_h1n1, Y_h3n2), axis = 0)
	XNameTrain2 = np.concatenate((XName_h1n1, XName_h3n2), axis = 0)

	Xtest2 = np.concatenate((X_h5n1, X_h9n2), axis = 0)
	Ytest2 = np.concatenate((Y_h5n1, Y_h9n2), axis = 0)
	XNameTest2 = np.concatenate((XName_h5n1, XName_h9n2), axis = 0)

	labels_train2 = create_binary_labels(Ytrain2, 4)
	labels_test2 = create_binary_labels(Ytest2, 4)

	for i in range(10):
		outfile = "../result/srep2017Peng/tl2_" + str(i) + ".nb"
		X_train, Y_train, labels_train, XName_train = shuffle_dataset_with_labels(Xtrain2, Ytrain2, labels_train2, XNameTrain2)
		X_test, Y_test, labels_test, XName_test = shuffle_dataset_with_labels(Xtest2, Ytest2, labels_test2, XNameTest2)

		result = naiveBayes(outfile, X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test)


def transfer_learning(test_data):
	# print "transfer learning test on: " + test_data

	dh1n1, dh3n2, dh5n1, dh9n2, dmixed = mix_subtypes_dist()

	h1n1_fm_file = featureMatrix_path + "h1n1.feats"
	h3n2_fm_file = featureMatrix_path + "h3n2.feats"
	h5n1_fm_file = featureMatrix_path + "h5n1.feats"
	h9n2_fm_file = featureMatrix_path + "h9n2.feats"
	
	X_h1n1, Y_h1n1, XName_h1n1 = load_features_dataset(h1n1_fm_file, dh1n1)
	X_h3n2, Y_h3n2, XName_h3n2 = load_features_dataset(h3n2_fm_file, dh3n2)
	X_h5n1, Y_h5n1, XName_h5n1 = load_features_dataset(h5n1_fm_file, dh5n1)
	X_h9n2, Y_h9n2, XName_h9n2 = load_features_dataset(h9n2_fm_file, dh9n2)

	dataset = {}
	dataset["h1n1"] = (X_h1n1, Y_h1n1, XName_h1n1)
	dataset["h3n2"] = (X_h3n2, Y_h3n2, XName_h3n2)
	dataset["h5n1"] = (X_h5n1, Y_h5n1, XName_h5n1)
	dataset["h9n2"] = (X_h9n2, Y_h9n2, XName_h9n2)

	tmp = ["h1n1", "h3n2", "h5n1", "h9n2"]
	tmp.remove(test_data)

	(X_test, Y_test, XName_test) = dataset[test_data]
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)
	XName_test = np.array(XName_test)
	labels_test = create_binary_labels(Y_test, 4)

	(X1, Y1, XName1) = dataset[tmp[0]]
	(X2, Y2, XName2) = dataset[tmp[1]]
	(X3, Y3, XName3) = dataset[tmp[2]]
	X_train = np.concatenate((X1, X2, X3), axis = 0)
	Y_train = np.concatenate((Y1, Y2, Y3), axis = 0)
	XName_train = np.concatenate((XName1, XName2, XName3), axis = 0)
	labels_train = create_binary_labels(Y_train, 4)

	for i in range(10):
		outfile = "../result/srep2017Peng/tl_" + test_data + "_" + str(i) + ".nb"
		X_train, Y_train, labels_train, XName_train = shuffle_dataset_with_labels(X_train, Y_train, labels_train, XName_train)
		X_test, Y_test, labels_test, XName_test = shuffle_dataset_with_labels(X_test, Y_test, labels_test, XName_test)
		result = naiveBayes(outfile, X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test)

def writeResults_csv():
	sub_prefix = ["h1n1", "h3n2", "h5n1", "h9n2", "mixed"]
	trans_prefix = ["tl_h1n1", "tl_h3n2", "tl_h5n1", "tl_h9n2", "tl2"]

	prefixList = sub_prefix + trans_prefix

	result_csv = open("../result/srep2017Peng/srep2017Peng.csv", "w")
	result_csv.write("Dataset, TrainAccu, TestAccu, TrainPrec, TestPrec, TrainRec, TestRec, TrainF1, TestF1\n")

	path = "../result/srep2017Peng/"

	for prefix in prefixList:		

		train_rmse_list = []
		train_accu_list = []
		train_prec_list = []
		train_rec_list = []
		train_f1_list = []

		test_rmse_list = []
		test_accu_list = []
		test_prec_list = []
		test_rec_list = []
		test_f1_list = []

		for i in range(10):
			with open(path+prefix+"_"+str(i)+".nb") as f:
				result = pickle.load(f)

			train_accu_list.append(result["train_accu"])
			train_prec_list.append(result["train_prec"])
			train_rec_list.append(result["train_rec"])
			train_f1_list.append(result["train_f1"])

			test_accu_list.append(result["test_accu"])
			test_prec_list.append(result["test_prec"])
			test_rec_list.append(result["test_rec"])
			test_f1_list.append(result["test_f1"])

		avg_accu1 = np.array(train_accu_list).mean()
		avg_accu2 = np.array(test_accu_list).mean()

		avg_prec1 = np.array(train_prec_list).mean()
		avg_prec2 = np.array(test_prec_list).mean()

		avg_rec1 = np.array(train_rec_list).mean()
		avg_rec2 = np.array(test_rec_list).mean()

		avg_f1_1 = np.array(train_f1_list).mean()
		avg_f1_2 = np.array(test_f1_list).mean()

		line = prefix + ", " + str(avg_accu1) + ", " + str(avg_accu2) + ", "
		line = line + str(avg_prec1) + ", " + str(avg_prec2) + ", "
		line = line + str(avg_rec1) + ", " + str(avg_rec2) + ", "
		line = line + str(avg_f1_1) + ", " + str(avg_f1_2) + "\n"

		result_csv.write(line)

	result_csv.close()



if __name__ == '__main__':

	write_feature_matrix()

	seqs_H1N1, seqs_H3N2, seqs_H5N1, seqs_H9N2, mixed_seqs = load_aligned_seqs()
	dh1n1, dh3n2, dh5n1, dh9n2, dmixed = mix_subtypes_dist()

	trainNB("h1n1", dh1n1)
	trainNB("h3n2", dh3n2)
	trainNB("h5n1", dh5n1)
	trainNB("h9n2", dh9n2)
	trainNB("mixed", dmixed)

	transfer_learning_2()
	transfer_learning("h1n1")
	transfer_learning("h3n2")
	transfer_learning("h5n1")
	transfer_learning("h9n2")

	writeResults_csv()

