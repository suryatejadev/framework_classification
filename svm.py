# inbuilt modules
import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm
# manual modules
from dstML.eval_metrics import *
from dstML.data_fn import *

if __name__=='__main__':
	
    # load data
    data_dict = loadmat('data/oct_data.mat')
    data,label = data_dict['data'],data_dict['label'][0,:]

    # model parameters 
    nb_classes = 2
    lr = 1e-3
    total_iter = 2
    n_roc = 55
    split = 0.7
    conf_matrix_list_train = []
    conf_matrix_list_test = []
    # number of times data is randomized
    for n_iter in tqdm(range(total_iter)):
        
        # get training, testing data with data balancingi
        x_train,x_test,y_train,y_test = split_data(data,label,split,balance_data=True)
        # prepare the data for efficient modelling
        x_train,x_test = prepare_data(x_train,x_test)
        # train the network !
        clf = svm.SVC(gamma=0.1,C=200,kernel='rbf',verbose=False)
        clf = clf.fit(x_train,y_train)
        conf_matrix_list_train.append(confmatrix_svm(clf,x_train,y_train,n_roc,pos_label=0))
        conf_matrix_list_test.append(confmatrix_svm(clf,x_test,y_test,n_roc,pos_label=0))

    # get evaluation scores mean and std
    eval_score_train = eval_metric_stats(conf_matrix_list_train)
    eval_score_test = eval_metric_stats(conf_matrix_list_test)   
    print 'training scores : '
    print 'acc = ',round(eval_score_train['acc'][0],3),' ',round(eval_score_train['acc'][1],3)
    print 'tpr = ',round(eval_score_train['tpr'][0],3),' ',round(eval_score_train['tpr'][1],3)
    print 'tnr = ',round(eval_score_train['tnr'][0],3),' ',round(eval_score_train['tnr'][1],3)
    print 'auc = ',round(eval_score_train['auc'][0],3),' ',round(eval_score_train['auc'][1],3)
    print 'testing scores : '
    print 'acc = ',round(eval_score_test['acc'][0],3),' ',round(eval_score_test['acc'][1],3)
    print 'tpr = ',round(eval_score_test['tpr'][0],3),' ',round(eval_score_test['tpr'][1],3)
    print 'tnr = ',round(eval_score_test['tnr'][0],3),' ',round(eval_score_test['tnr'][1],3)
    print 'auc = ',round(eval_score_test['auc'][0],3),' ',round(eval_score_test['auc'][1],3)
       
