from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from utils import load_dataset, create_binary_labels, shuffle_dataset_with_labels, avoid_none, split_train_test_with_labels

from sklearn.ensemble import RandomForestRegressor
from math import sqrt

import pickle
import numpy as np
import warnings



def rf_model(outfile, X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test):
	"""
	"""

	regr = RandomForestRegressor(max_depth = 9)
	regr_model = regr.fit(X_train, Y_train)

	ytrain_regr = regr_model.predict(X_train)
	ytrain_regr_label = create_binary_labels(ytrain_regr, 4)

	ytest_regr = regr_model.predict(X_test)
	ytest_regr_label = create_binary_labels(ytest_regr, 4)

	ytrain_regr_label = ytrain_regr_label.tolist()
	labels_train = labels_train.tolist()

	train_rmse = sqrt(mean_squared_error(Y_train, ytrain_regr))
	train_accuracy = accuracy_score(labels_train, ytrain_regr_label)
	train_precision = precision_score(labels_train, ytrain_regr_label)
	train_recall = recall_score(labels_train, ytrain_regr_label)
	train_f1 = f1_score(labels_train, ytrain_regr_label)

	test_rmse = sqrt(mean_squared_error(Y_test, ytest_regr))
	test_accuracy = accuracy_score(labels_test, ytest_regr_label)
	test_precision = precision_score(labels_test, ytest_regr_label)
	test_recall = recall_score(labels_test, ytest_regr_label)
	test_f1 = f1_score(labels_test, ytest_regr_label)

	feature_importance = regr.feature_importances_

	result = {}
	result["rf_model"] = regr_model
	result["feature_importance"] = feature_importance

	result["train_rmse"] = train_rmse
	result["test_rmse"] = test_rmse
	
	result["train_accu"] = train_accuracy
	result["test_accu"] = test_accuracy

	result["train_prec"] = train_precision
	result["test_prec"] = test_precision
	
	result["train_rec"] = train_recall
	result["test_rec"] = test_recall

	result["train_f1"] = train_f1
	result["test_f1"] = test_f1

	with open(outfile, "w") as f:
		pickle.dump(result, f)

	return result

def kfold_trainRF(k, indexID, prefix, X, Y, labels, XName):
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
	for i in range(k):
		outfile = "../result/randomForest/" + prefix + "_" + indexID + "_" + str(i) + ".rf"
		X, Y, labels, XName = shuffle_dataset_with_labels(X, Y, labels, XName)
		X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test = split_train_test_with_labels(X, Y, labels, XName, 0.8)
		result = rf_model(outfile, X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test)

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
	avg_rmse2 = np.array(test_rmse_list).mean()
	avg_accu1 = np.array(train_accu_list).mean()
	avg_accu2 = np.array(test_accu_list).mean()
	avg_prec1 = np.array(train_prec_list).mean()
	avg_prec2 = np.array(test_prec_list).mean()
	avg_rec1 = np.array(train_rec_list).mean()
	avg_rec2 = np.array(test_rec_list).mean()
	avg_f1_1 = np.array(train_f1_list).mean()
	avg_f1_2 = np.array(test_f1_list).mean()

	print "*** " + prefix + " ****" + indexID + "****"
	print "avg_rmse1: " + str(avg_rmse1) + "\t avg_rmse2: " + str(avg_rmse2)
	print "avg_accu1: " + str(avg_accu1) + "\t avg_accu2: " + str(avg_accu2)
	print "avg_prec1: " + str(avg_prec1) + "\t avg_prec2: " + str(avg_prec2)
	print "avg_rec1: " + str(avg_rec1) + "\t avg_rec2: " + str(avg_rec2)
	print "avg_f1_1: " + str(avg_f1_1) + "\t avg_f1_2: " + str(avg_f1_2)
	print " "

def rf_cross_validation(indexID, prefix):
	dataset = load_dataset(indexID)
	(X, Y, XName) = dataset[prefix]
	X, Y = avoid_none(X, Y, 15)
	labels = create_binary_labels(Y, 4)

	kfold_trainRF(10, indexID, prefix, X, Y, labels, XName)

# rf_cross_validation("ALTS910101", "h1n1")

def rf_transfer_learning(indexID):
	dataset = load_dataset(indexID)

	(X_h1n1, Y_h1n1, XName_h1n1) = dataset["h1n1"]
	(X_h3n2, Y_h3n2, XName_h3n2) = dataset["h3n2"]
	(X_h5n1, Y_h5n1, XName_h5n1) = dataset["h5n1"]
	(X_h9n2, Y_h9n2, XName_h9n2) = dataset["h9n2"]

	X_h1n1, Y_h1n1 = avoid_none(X_h1n1, Y_h1n1, 15)
	X_h3n2, Y_h3n2 = avoid_none(X_h3n2, Y_h3n2, 15)
	X_h5n1, Y_h5n1 = avoid_none(X_h5n1, Y_h5n1, 15)
	X_h9n2, Y_h9n2 = avoid_none(X_h9n2, Y_h9n2, 15)

	X_h1n1 = X_h1n1.reshape(X_h1n1.shape[0], -1)
	X_h3n2 = X_h3n2.reshape(X_h3n2.shape[0], -1)
	X_h5n1 = X_h5n1.reshape(X_h5n1.shape[0], -1)
	X_h9n2 = X_h9n2.reshape(X_h9n2.shape[0], -1)

	h1n1_labels = create_binary_labels(Y_h1n1, 4)
	h3n2_labels = create_binary_labels(Y_h3n2, 4)
	h5n1_labels = create_binary_labels(Y_h5n1, 4)
	h9n2_labels = create_binary_labels(Y_h9n2, 4)

	# Train on H1N1+H3N2;l Test on H5N1+H9N2
	Xtrain2 = np.concatenate((X_h1n1, X_h3n2), axis = 0)
	Ytrain2 = np.concatenate((Y_h1n1, Y_h3n2), axis = 0)
	XName_train2 = np.concatenate((XName_h1n1, XName_h3n2), axis = 0)

	Xtest2 = np.concatenate((X_h5n1, X_h9n2), axis = 0)
	Ytest2 = np.concatenate((Y_h5n1, Y_h9n2), axis = 0)
	XName_test2 = np.concatenate((XName_h5n1, XName_h9n2), axis = 0)

	Xtrain2, Ytrain2 = avoid_none(Xtrain2, Ytrain2, 15)
	Xtest2, Ytest2 = avoid_none(Xtest2, Ytest2, 15)

	labels_train2 = create_binary_labels(Ytrain2, 4)
	labels_test2 = create_binary_labels(Ytest2, 4)

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
		outfile = "../result/randomForest/tl2_" + indexID + "_" + str(i) + ".rf"
		X_train, Y_train, labels_train, XName_train = shuffle_dataset_with_labels(Xtrain2, Ytrain2, labels_train2, XName_train2)
		X_test, Y_test, labels_test, XName_test = shuffle_dataset_with_labels(Xtest2, Ytest2, labels_test2, XName_test2)
		result = rf_model(outfile, X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test)

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
	avg_rmse2 = np.array(test_rmse_list).mean()
	avg_accu1 = np.array(train_accu_list).mean()
	avg_accu2 = np.array(test_accu_list).mean()
	avg_prec1 = np.array(train_prec_list).mean()
	avg_prec2 = np.array(test_prec_list).mean()
	avg_rec1 = np.array(train_rec_list).mean()
	avg_rec2 = np.array(test_rec_list).mean()
	avg_f1_1 = np.array(train_f1_list).mean()
	avg_f1_2 = np.array(test_f1_list).mean()

	print "*** Transfer learning ** Train on H1N1+H3N2; Test on H5N1+H9N2 **** " + indexID + "****"
	print "avg_rmse1: " + str(avg_rmse1) + "\t avg_rmse2: " + str(avg_rmse2)
	print "avg_accu1: " + str(avg_accu1) + "\t avg_accu2: " + str(avg_accu2)
	print "avg_prec1: " + str(avg_prec1) + "\t avg_prec2: " + str(avg_prec2)
	print "avg_rec1: " + str(avg_rec1) + "\t avg_rec2: " + str(avg_rec2)
	print "avg_f1_1: " + str(avg_f1_1) + "\t avg_f1_2: " + str(avg_f1_2)
	print " "

	# Train on any three datasets and test on the rest
	names = ["h1n1", "h3n2", "h5n1", "h9n2"]
	for name in names:
		(X_test, Y_test, XName_test) = dataset[name]
		X_test = X_test.reshape(X_test.shape[0], -1)
		
		tmp = ["h1n1", "h3n2", "h5n1", "h9n2"]
		tmp.remove(name)

		(X1, Y1, XName1) = dataset[tmp[0]]
		(X2, Y2, XName2) = dataset[tmp[1]]
		(X3, Y3, XName3) = dataset[tmp[2]]

		X1 = X1.reshape(X1.shape[0], -1)
		X2 = X2.reshape(X2.shape[0], -1) 
		X3 = X3.reshape(X3.shape[0], -1)

		X_train = np.concatenate((X1, X2, X3), axis = 0)
		Y_train = np.concatenate((Y1, Y2, Y3), axis = 0)
		XName_train = np.concatenate((XName1, XName2, XName3), axis = 0)

		X_train, Y_train = avoid_none(X_train, Y_train, 15)
		X_test, Y_test = avoid_none(X_test, Y_test, 15)

		labels_train = create_binary_labels(Y_train, 4)
		labels_test = create_binary_labels(Y_test, 4)

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
			outfile = "../result/randomForest/tl_" + name +"_" + indexID + "_" + str(i) + ".rf"
			X_train, Y_train, labels_train, XName_train = shuffle_dataset_with_labels(X_train, Y_train, labels_train, XName_train)
			X_test, Y_test, labels_test, XName_test = shuffle_dataset_with_labels(X_test, Y_test, labels_test, XName_test)

			result = rf_model(outfile, X_train, Y_train, labels_train, XName_train, X_test, Y_test, labels_test, XName_test)

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
		avg_rmse2 = np.array(test_rmse_list).mean()
		avg_accu1 = np.array(train_accu_list).mean()
		avg_accu2 = np.array(test_accu_list).mean()
		avg_prec1 = np.array(train_prec_list).mean()
		avg_prec2 = np.array(test_prec_list).mean()
		avg_rec1 = np.array(train_rec_list).mean()
		avg_rec2 = np.array(test_rec_list).mean()
		avg_f1_1 = np.array(train_f1_list).mean()
		avg_f1_2 = np.array(test_f1_list).mean()

		print "*** Transfer learning *** Test @: " + name + " ****** " + indexID + " **"
		print "avg_rmse1: " + str(avg_rmse1) + "\t avg_rmse2: " + str(avg_rmse2)
		print "avg_accu1: " + str(avg_accu1) + "\t avg_accu2: " + str(avg_accu2)
		print "avg_prec1: " + str(avg_prec1) + "\t avg_prec2: " + str(avg_prec2)
		print "avg_rec1: " + str(avg_rec1) + "\t avg_rec2: " + str(avg_rec2)
		print "avg_f1_1: " + str(avg_f1_1) + "\t avg_f1_2: " + str(avg_f1_2)
		print " "
