import numpy as np

from sklearn.model_selection import KFold
from dsbase.ModelDSBase import ModelDSBaseWrapper

class KFoldDSBase:

	def __init__(self, X, y, k, model_class, model_prefix_name, parameters):
		self.models = []
		self.scores_train = []
		self.scores_test = []
		self.X = X
		self.y = y
		self.k = k
		self.model_class = model_class
		self.model_prefix_name = model_prefix_name
		self.parameters = parameters

	def train(self):
		kf = KFold(n_splits=self.k)
		self.models = []
		self.scores = []
		index = 1
		for train_index, test_index in kf.split(self.X):
		    #print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
		    print('---- Model ',index,'-------------------------------------')
		    X_train, X_test = self.X[train_index], self.X[test_index]
		    y_train, y_test = self.y[train_index], self.y[test_index]
		    model = ModelDSBaseWrapper(self.model_prefix_name, X_train, y_train, X_test, y_test,[100], self.model_class, self.parameters)
		    model.train()
		    lcmodel=model.getLearningCurves()
		    print('-> train score:',lcmodel[0,-1],'/ test score:',lcmodel[1,-1])
		    self.models.append(model)
		    self.scores_train.append(lcmodel[0,-1])
		    self.scores_test.append(lcmodel[1,-1])
		    index+=1
		    print('---------------------------------------------------')
		print('Avg Score:',np.mean(self.scores))

	def getMeanScore(self):
		return (np.mean(self.scores_train), np.mean(self.scores_test))

	def getBestModel(self):
		return self.models[np.argmax(self.scores_test)].model
