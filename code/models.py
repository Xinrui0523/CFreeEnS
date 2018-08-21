from sklearn import ensemble
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, learning_curve

from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score, f1_score, precision_score, recall_score
# from imblearn.metrics import geometric_mean_score
from plot_learning_curve import data_size_response, plot_response

import warnings
import numpy as np

warnings.filterwarnings("ignore")

def rf_regr(x, y, fname):
    regr = ensemble.RandomForestRegressor(max_depth = 9)

    # x_train 70% of the set 
    # And a 15% / 15% split on CV and Test.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # x_cv, x_test, y_cv, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=10)

    regr_model = regr.fit(x_train, y_train)
    y_test_pred = regr_model.predict(x_test)

    y_pred_labels = (y_test_pred > 4)
    y_test_labels = (y_test > 4)

    accuracy = accuracy_score(y_pred_labels, y_test_labels)
    precision = precision_score(y_pred_labels, y_test_labels)
    recall = recall_score(y_pred_labels, y_test_labels)
    f1 = f1_score(y_pred_labels, y_test_labels)

    subset_sizes,train_errs,test_errs = data_size_response(regr, x_train, x_test, y_train, y_test, 30)
    # print('Plotting...')
    plot_response(fname, subset_sizes,train_errs,test_errs)

    return accuracy, precision, recall, f1


def rf_regr_avg(x, y, k, prefix):
    accu_list = []
    prec_list = []
    recall_list = []
    f1_list = []

    for i in range(k):
        fname = '../result/' + prefix + '_' + str(i) + '.svg'
        accuracy, precision, recall, f1 = rf_regr(x, y, fname)

        accu_list.append(accuracy)
        prec_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    avg_accu = np.array(accu_list).mean()
    avg_prec = np.array(prec_list).mean()
    avg_rec = np.array(recall_list).mean()
    avg_f1 = np.array(f1_list).mean()

    return avg_accu, avg_prec, avg_rec, avg_f1

    
def rf_cv(x, y):
    # np.random.seed(313)
    clf = ensemble.RandomForestClassifier(max_depth = 9)
    
    accuracy = cross_val_score(clf, x, y, cv = 10, scoring='accuracy')
    print("Accuracy: %f" % accuracy.mean())
    
    precision = cross_val_score(clf, x, y, cv = 10, scoring ='precision_macro')
    recall = cross_val_score(clf, x, y, cv = 10, scoring = 'recall_macro')
    f_measure = cross_val_score(clf, x, y, cv = 10, scoring = 'f1_macro')
    
    y_pred = cross_val_predict(clf, x, y, cv = 10)
    # g_mean = geometric_mean_score(y, y_pred)
    # print("g_mean: %f " % g_mean.mean())
    
    mcc = matthews_corrcoef(y, y_pred)
    # print("MCC: %f" % mcc)
    
    classification_rep = classification_report(y, y_pred)
    
    print("Classification report: " )
    print(classification_rep)

