from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import mix_subtypes_dist, create_binary_labels
from utils import shuffle_dataset_with_labels, split_train_test_with_labels
from rf_model import rf_model
from srep2017Peng import load_aligned_seqs, load_features_dataset

import numpy as np
import pickle
import aaindex

from Bio import SeqIO
from heapq import nlargest
from itertools import compress
from sklearn.ensemble import RandomForestRegressor

featureMatrix_path = "../data/srep2017Peng_featMatrix/"

def kfold_srep(k, prefix, X, Y, XName):
	for i in range(k):
		outfile = "../result/srep2017Adapted/" + prefix + "_" + str(i) + ".rf_srep"
		labels = create_binary_labels(Y, 4)

		X = np.array(X)
		Y = np.array(Y).reshape(len(Y),1)
		XName = np.array(XName).reshape(len(XName),1)
		labels = np.array(labels)

		X, Y, labels, XName = shuffle_dataset_with_labels(X, Y, labels, XName)
		Xtrain, Ytrain, labelsTrain, XNameTrain, Xtest, Ytest, labelsTest, XNameTest = split_train_test_with_labels(X, Y, labels, XName, 0.8)
		result = rf_model(outfile, Xtrain, Ytrain, labelsTrain, XNameTrain, Xtest, Ytest, labelsTest, XNameTest)


def train_rfsrep(prefix, dist_dict):
	featureMatrix_file = featureMatrix_path + prefix + ".feats"
	X, Y, XName = load_features_dataset(featureMatrix_file, dist_dict)

	kfold_srep(10, prefix, X, Y, XName)

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
		outfile = "../result/srep2017Adapted/tl2_" + str(i) + ".rf_srep"
		X_train, Y_train, labels_train, XName_train = shuffle_dataset_with_labels(Xtrain2, Ytrain2, labels_train2, XNameTrain2)
		X_test, Y_test, labels_test, XName_test = shuffle_dataset_with_labels(Xtest2, Ytest2, labels_test2, XNameTest2)

		result = rf_model(outfile, X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test)


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
		outfile = "../result/srep2017Adapted/tl_" + test_data + "_" + str(i) + ".rf_srep"
		X_train, Y_train, labels_train, XName_train = shuffle_dataset_with_labels(X_train, Y_train, labels_train, XName_train)
		X_test, Y_test, labels_test, XName_test = shuffle_dataset_with_labels(X_test, Y_test, labels_test, XName_test)
		result = rf_model(outfile, X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test)


def writeResults_csv():
	sub_prefix = ["h1n1", "h3n2", "h5n1", "h9n2", "mixed"]
	trans_prefix = ["tl_h1n1", "tl_h3n2", "tl_h5n1", "tl_h9n2", "tl2"]

	prefixList = sub_prefix + trans_prefix

	result_csv = open("../result/srep2017Adapted/srep2017Adapted.csv", "w")
	result_csv.write("Dataset, TrainAccu, TestAccu, TrainPrec, TestPrec, TrainRec, TestRec, TrainF1, TestF1\n")

	path = "../result/srep2017Adapted/"

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
			with open(path+prefix+"_"+str(i)+".rf_srep") as f:
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

	seqs_H1N1, seqs_H3N2, seqs_H5N1, seqs_H9N2, mixed_seqs = load_aligned_seqs()
	dh1n1, dh3n2, dh5n1, dh9n2, dmixed = mix_subtypes_dist()

	train_rfsrep("h1n1", dh1n1)
	train_rfsrep("h3n2", dh3n2)
	train_rfsrep("h5n1", dh5n1)
	train_rfsrep("h9n2", dh9n2)
	train_rfsrep("mixed", dmixed)

	transfer_learning_2()
	transfer_learning("h1n1")
	transfer_learning("h3n2")
	transfer_learning("h5n1")
	transfer_learning("h9n2")

	writeResults_csv()





