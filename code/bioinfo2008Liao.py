from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from utils import load_aligned_seqs, mix_subtypes_dist, create_binary_labels
from utils import shuffle_dataset_with_labels, split_train_test_with_labels
from math import sqrt

import statsmodels.api as sm
import numpy as np
import pickle

mutMatrix_path = "../data/bioinfo2008Liao_mutMatrix/"

def get_mutations(s1, s2):
	"""
	mutations are flagged as 1 (True)
	"""
	assert len(s1) == len(s2)
	mutations = list(i[0] != i[1] for i in zip(s1, s2))
	mutations = np.array(mutations).astype(int)

	return mutations

def boolen_GM5(a, b):
	g1 = ["A", "I", "L", "M", "P", "V"]
	g2 = ["F", "W", "Y"]
	g3 = ["N", "Q", "S", "T"]
	g4 = ["H", "K", "R"]
	g5 = ["D", "E"]
	g6 = ["C"]
	g7 = ["G"]
	g8 = ["-"]

	same1 = (a in g1) and (b in g1)
	same2 = (a in g2) and (b in g2)
	same3 = (a in g3) and (b in g3)
	same4 = (a in g4) and (b in g4)
	same5 = (a in g5) and (b in g5)
	same6 = (a in g6) and (b in g6)
	same7 = (a in g7) and (b in g7)
	same8 = (a in g8) and (b in g8)

	flag = same1 or same2 or same3 or same4 or same5 or same6 or same7 or same8

	return flag


def get_mutations_GM5(s1, s2):
	"""
	GM5: 
	{non-polar aliphatic: A, I, L, M, P, V}, 
	{non-polar aromatic: F, W, Y}, 
	{polar: N, Q, S, T}, 
	{positively charged: H, K, R}, 
	{negatively charged: D, E}, 
	{C}, 
	{G} 
	(Espadaler et al., 2005)

	Amino acid residues change 
		between grous --> 1
		within the same group --> 0. 
	"""
	assert len(s1) == len(s2)
	mutations = []
	for i in range(len(s1)):
		flag = boolen_GM5(s1[i], s2[i])
		mutations.append(flag)

	mutations = np.array(mutations).astype(int)
	return mutations


def seqList2Dict(seqs_list):
	"""
	"""
	seqs_dict = {}
	for s in seqs_list:
		vname = s.id
		vseq = s.seq
		seqs_dict[vname] = vseq
	return seqs_dict



def get_mutMatrix(seqs_list, dist_dict, outfile):
	"""
	"""
	XNamelist = dist_dict.keys()
	YList = dist_dict.values()

	seqs_dict = seqList2Dict(seqs_list)

	mutMatrix_dict = {}
	for name in XNamelist:
		vnames = name.split(",")
		v1 = vnames[0]
		v2 = vnames[1]
		s1 = seqs_dict[v1]
		s2 = seqs_dict[v2]

		# mutMatrix_dict[v1 + "," + v2] = get_mutations(s1, s2)
		mutMatrix_dict[v1 + "," + v2] = get_mutations_GM5(s1, s2)

	with open(outfile, "w") as f:
		pickle.dump(mutMatrix_dict, f)

	return mutMatrix_dict

def writeMutMatrix():
	seqs_h1n1, seqs_h3n2, seqs_h5n1, seqs_h9n2, mixed_seqs = load_aligned_seqs()
	dh1n1, dh3n2, dh5n1, dh9n2, dmixed = mix_subtypes_dist()

	# mutMatrix_path = "../data/bioinfo2008Liao_mutMatrix/"
	h1n1_mutMatrix = get_mutMatrix(seqs_h1n1, dh1n1, mutMatrix_path + "h1n1.mut")
	h3n2_mutMatrix = get_mutMatrix(seqs_h3n2, dh3n2, mutMatrix_path + "h3n2.mut")
	h5n1_mutMatrix = get_mutMatrix(seqs_h5n1, dh5n1, mutMatrix_path + "h5n1.mut")
	h9n2_mutMatrix = get_mutMatrix(seqs_h9n2, dh9n2, mutMatrix_path + "h9n2.mut")
	mixed_mutMatrix = get_mutMatrix(mixed_seqs, dmixed, mutMatrix_path + "mixed.mut")

def load_mut_dataset(mutMatrix_file, dist_dict):
	with open(mutMatrix_file, "r") as f:
		mutMatrix = pickle.load(f)
	X = []
	Y = []
	XName = []

	for vname in dist_dict.keys():
		X.append(mutMatrix[vname])
		Y.append(dist_dict[vname])
		Y = [float(i) for i in Y]
		XName.append(vname)

	return X, Y, XName

def multiReg_GM5(outfile, Xtrain, Ytrain, labelsTrain, XNameTrain, Xtest, Ytest, labelsTest, XNameTest):
	"""
	"""

	model = sm.OLS(Ytrain, Xtrain).fit()
	Ytrain_est = model.predict(Xtrain)
	labels_train_est = create_binary_labels(Ytrain_est, 4)

	Ytest_est = model.predict(Xtest)
	labels_test_est = create_binary_labels(Ytest_est, 4)

	train_rmse = sqrt(mean_squared_error(Ytrain, Ytrain_est))
	train_accu = accuracy_score(labelsTrain, labels_train_est)
	train_prec = precision_score(labelsTrain, labels_train_est)
	train_rec = recall_score(labelsTrain, labels_train_est)
	train_f1 = f1_score(labelsTrain, labels_train_est)

	test_rmse = sqrt(mean_squared_error(Ytest, Ytest_est))
	test_accu = accuracy_score(labelsTest, labels_test_est)
	test_prec = precision_score(labelsTest, labels_test_est)
	test_rec = recall_score(labelsTest, labels_test_est)
	test_f1 = f1_score(labelsTest, labels_test_est)

	result = {}
	result["model"] = model

	result["train_rmse"] = train_rmse
	result["train_accu"] = train_accu
	result["train_prec"] = train_prec
	result["train_rec"] = train_rec
	result["train_f1"] = train_f1

	result["test_rmse"] = test_rmse
	result["test_accu"] = test_accu
	result["test_prec"] = test_prec
	result["test_rec"] = test_rec
	result["test_f1"] = test_f1

	with open(outfile, "w") as f:
		pickle.dump(result, f)

	return result

def kfold_trainGM5(k, prefix, X, Y, XName):

	for i in range(k):
		outfile = "../result/bioinfo2008Liao/" + prefix + "_" + str(i) + ".gm5"
		labels = create_binary_labels(Y, 4)
		
		X = np.array(X)
		Y = np.array(Y).reshape(len(Y),1)
		XName = np.array(XName).reshape(len(XName),1)
		labels = np.array(labels)

		X, Y, labels, XName = shuffle_dataset_with_labels(X, Y, labels, XName)
		Xtrain, Ytrain, labelsTrain, XNameTrain, Xtest, Ytest, labelsTest, XNameTest = split_train_test_with_labels(X, Y, labels, XName, 0.8)
		result = multiReg_GM5(outfile, Xtrain, Ytrain, labelsTrain, XNameTrain, Xtest, Ytest, labelsTest, XNameTest)

def trainGM5(prefix, dist_dict):
	
	mutMatrix_file = mutMatrix_path + prefix + ".mut"
	X, Y, XName = load_mut_dataset(mutMatrix_file, dist_dict)

	kfold_trainGM5(10, prefix, X, Y, XName)

	pass

def transfer_learning_2():
	dh1n1, dh3n2, dh5n1, dh9n2, dmixed = mix_subtypes_dist()

	h1n1_mutMatrix_file = mutMatrix_path + "h1n1.mut"
	h3n2_mutMatrix_file = mutMatrix_path + "h3n2.mut"
	h5n1_mutMatrix_file = mutMatrix_path + "h5n1.mut"
	h9n2_mutMatrix_file = mutMatrix_path + "h9n2.mut"
	X_h1n1, Y_h1n1, XName_h1n1 = load_mut_dataset(h1n1_mutMatrix_file, dh1n1)
	X_h3n2, Y_h3n2, XName_h3n2 = load_mut_dataset(h3n2_mutMatrix_file, dh3n2)
	X_h5n1, Y_h5n1, XName_h5n1 = load_mut_dataset(h5n1_mutMatrix_file, dh5n1)
	X_h9n2, Y_h9n2, XName_h9n2 = load_mut_dataset(h9n2_mutMatrix_file, dh9n2)

	dataset = {}
	dataset["h1n1"] = (X_h1n1, Y_h1n1, XName_h1n1)
	dataset["h3n2"] = (X_h3n2, Y_h3n2, XName_h3n2)
	dataset["h5n1"] = (X_h5n1, Y_h5n1, XName_h5n1)
	dataset["h9n2"] = (X_h9n2, Y_h9n2, XName_h9n2)

	# Train on H1N1+H3N2; Test on H5N1+H9N2
	Xtrain2 = np.concatenate((X_h1n1, X_h3n2), axis = 0)
	Ytrain2 = np.concatenate((Y_h1n1, Y_h3n2), axis = 0)
	XNameTrain2 = np.concatenate((XName_h1n1, XName_h3n2), axis = 0)

	Xtest2 = np.concatenate((X_h5n1, X_h9n2), axis = 0)
	Ytest2 = np.concatenate((Y_h5n1, Y_h9n2), axis = 0)
	XNameTest2 = np.concatenate((XName_h5n1, XName_h9n2), axis = 0)

	labels_train2 = create_binary_labels(Ytrain2, 4)
	labels_test2 = create_binary_labels(Ytest2, 4)

	for i in range(10):
		outfile = "../result/bioinfo2008Liao/tl2_" + str(i) + ".gm5"
		X_train, Y_train, labels_train, XName_train = shuffle_dataset_with_labels(Xtrain2, Ytrain2, labels_train2, XNameTrain2)
		X_test, Y_test, labels_test, XName_test = shuffle_dataset_with_labels(Xtest2, Ytest2, labels_test2, XNameTest2)

		result = multiReg_GM5(outfile, X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test)

def transfer_learning(test_data):
	# Train on any three datasets and test on the rest
	# e.g. train on H1N1+H3N2+H5N1; test on H9N2
	print test_data

	dh1n1, dh3n2, dh5n1, dh9n2, dmixed = mix_subtypes_dist()

	h1n1_mutMatrix_file = mutMatrix_path + "h1n1.mut"
	h3n2_mutMatrix_file = mutMatrix_path + "h3n2.mut"
	h5n1_mutMatrix_file = mutMatrix_path + "h5n1.mut"
	h9n2_mutMatrix_file = mutMatrix_path + "h9n2.mut"
	X_h1n1, Y_h1n1, XName_h1n1 = load_mut_dataset(h1n1_mutMatrix_file, dh1n1)
	X_h3n2, Y_h3n2, XName_h3n2 = load_mut_dataset(h3n2_mutMatrix_file, dh3n2)
	X_h5n1, Y_h5n1, XName_h5n1 = load_mut_dataset(h5n1_mutMatrix_file, dh5n1)
	X_h9n2, Y_h9n2, XName_h9n2 = load_mut_dataset(h9n2_mutMatrix_file, dh9n2)

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
		outfile = "../result/bioinfo2008Liao/tl_" + test_data + "_" + str(i) + ".gm5"
		X_train, Y_train, labels_train, XName_train = shuffle_dataset_with_labels(X_train, Y_train, labels_train, XName_train)
		X_test, Y_test, labels_test, XName_test = shuffle_dataset_with_labels(X_test, Y_test, labels_test, XName_test)

		result = multiReg_GM5(outfile, X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test)


def writeResults_csv():

	sub_prefix = ["h1n1", "h3n2", "h5n1", "h9n2", "mixed"]
	trans_prefix = ["tl_h1n1", "tl_h3n2", "tl_h5n1", "tl_h9n2", "tl2"]

	prefixList = sub_prefix + trans_prefix

	result_csv = open("../result/bioinfo2008Liao/bioinfo2008Liao.csv", "w")
	result_csv.write("Dataset, TrainRMSE, TestRMSE, TrainAccu, TestAccu, TrainPrec, TestPrec, TrainRec, TestRec, TrainF1, TestF1\n")

	path = "../result/bioinfo2008Liao/"

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
			with open(path+prefix+"_"+str(i)+".gm5") as f:
				result = pickle.load(f)

			train_rmse_list.append(result["train_rmse"])
			train_accu_list.append(result["train_accu"])
			train_prec_list.append(result["train_prec"])
			train_rec_list.append(result["train_rec"])
			train_f1_list.append(result["train_f1"])

			test_rmse_list.append(result["test_rmse"])
			test_accu_list.append(result["test_accu"])
			test_prec_list.append(result["test_prec"])
			test_rec_list.append(result["test_rec"])
			test_f1_list.append(result["test_f1"])

		avg_rmse1 = np.array(train_rmse_list).mean()
		avg_rmse2 = np.array(test_rec_list).mean()

		avg_accu1 = np.array(train_accu_list).mean()
		avg_accu2 = np.array(test_accu_list).mean()

		avg_prec1 = np.array(train_prec_list).mean()
		avg_prec2 = np.array(test_prec_list).mean()

		avg_rec1 = np.array(train_rec_list).mean()
		avg_rec2 = np.array(test_rec_list).mean()

		avg_f1_1 = np.array(train_f1_list).mean()
		avg_f1_2 = np.array(test_f1_list).mean()

		line = prefix + ", " + str(avg_rmse1) + ", " + str(avg_rmse2) + ", "
		line = line + str(avg_accu1) + ", " + str(avg_accu2) + ", "
		line = line + str(avg_prec1) + ", " + str(avg_prec2) + ", "
		line = line + str(avg_rec1) + ", " + str(avg_rec2) + ", "
		line = line + str(avg_f1_1) + ", " + str(avg_f1_2) + "\n"

		result_csv.write(line)

	result_csv.close()



if __name__ == '__main__':
	# Train the model 

	seqs_h1n1, seqs_h3n2, seqs_h5n1, seqs_h9n2, mixed_seqs = load_aligned_seqs()
	dh1n1, dh3n2, dh5n1, dh9n2, dmixed = mix_subtypes_dist()

	trainGM5("h1n1", dh1n1)
	trainGM5("h3n2", dh3n2)
	trainGM5("h5n1", dh5n1)
	trainGM5("h9n2", dh9n2)
	trainGM5("mixed", dmixed)

	transfer_learning_2()
	transfer_learning("h1n1")
	transfer_learning("h3n2")
	transfer_learning("h5n1")
	transfer_learning("h9n2")

	######################
	
	writeResults_csv()

