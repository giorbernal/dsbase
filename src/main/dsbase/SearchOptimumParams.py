import numpy as np
import pandas as pd
import random as rnd

from dsbase.KFoldDSBase import KFoldDSBase

def evaluateParams(X, y, k_fold, model_class, model_prefix_name, params_list, num_tries):
	tries = {}
	for i in range(num_tries):
	    kf = KFoldDSBase(X, y, 5, model_class, model_prefix_name, params_list[i])
	    kf.train()
	    score_train, score_test = kf.getMeanScore()
	    print('****** Result try',i,':',score_train,' / ',score_test)
	    tries[i]=(score_train,score_test,kf.getBestModel())
	return tries


def randomElement(vector):
    return vector[rnd.randrange(0,len(vector))]

def showSearchOptimumHyperParametersReport(tries):
    for tr in tries:
        print(tr,':',tries[tr][0],'/',tries[tr][1],'(',tries[tr][1]/tries[tr][0],')')

def getColumnsWithLessValue(df_columns, feature_importance_vector, level):
	ser = pd.Series(feature_importance_vector).value_counts().sort_index()
	acc = 0
	for i in range(level+1):
		acc += ser.iloc[i]
	return ser,df_columns[feature_importance_vector.argsort()][0:acc]