import numpy as np
import matplotlib.pyplot as plt 
import ConstantsDSBase as constants
import keras
from sklearn.externals import joblib
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense

description='DNNClassificationKeras'

class DNNClassificationKerasDSBaseModel:
    def __init__(self, id, X_train, y_train, X_test, y_test, parameters):
        self.id=id
        self.parameters=parameters
        if (y_train.any() != None):
            print("initiating model " + str(self.id) + ". " + description);
 
            self.X_train=X_train
            self.X_test=X_test
            self.y_train=y_train
            self.y_test=y_test

            self.input_size=X_train.shape[1]
            if (max(y_train)==1):
                self.output_size=1
            else:
                self.output_size=max(y_train)

            self.layers=self.parameters['layers']
            self.alpha=self.parameters['alpha']
            self.beta1=self.parameters['beta1']            
            self.beta2=self.parameters['beta2']
            self.epsilon=self.parameters['epsilon']

            # Create and set the model
            self.model = Sequential()

            self.model.add(Dense(units=self.layers[0], activation='sigmoid', input_dim=self.input_size))
            for l in self.layers[1:]:
                self.model.add(Dense(units=l, activation='sigmoid'))
            if (self.output_size==1):
                self.model.add(Dense(units=self.output_size, activation='sigmoid'))
                self.model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=self.alpha, beta_1=self.beta1, beta_2=self.beta2, epsilon=self.epsilon), metrics=['accuracy'])
            else:
                self.model.add(Dense(units=self.output_size, activation='softmax'))
                self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=self.alpha, beta_1=self.beta1, beta_2=self.beta2, epsilon=self.epsilon), metrics=['accuracy'])

        else:
            print("initiating empty model " + str(self.id) + ". " + description);
        
    def train(self):
        self.batch_size=self.parameters['batch_size']
        epochs=self.parameters['epochs']
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=self.batch_size)

    def predict(self, test_data):
        return self.model.predict_classes(test_data)

    def getTrainScore(self):
        return self.model.evaluate(self.X_train, self.y_train, batch_size=self.batch_size)[1]
    
    def getTestScore(self):
        return self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)[1]
        
    def save(self, folder_path=constants.PERSISTANCE_FOLDER):
        file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT  # .h5 file
        self.model.save(file_path)
    
    def load(self, folder_path=constants.PERSISTANCE_FOLDER):
        file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT  # .h5 file
        self.model = load_model(file_path)

    def close(self):
        pass

    def __ifEmptyModel(self, vector):
        if ((vector.mean()==1) & (vector.var()==0)):
            return 1
        else:
            return 0

# Params converter function. Reference for every model
def DNNClassificationKerasDSBaseParamsToMap(layers, alpha=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8,batch_size=128, epochs=10):
    params={}
    params['layers']=layers
    params['alpha']=alpha
    params['beta1']=beta1
    params['beta2']=beta2
    params['epsilon']=epsilon       
    params['batch_size']=batch_size
    params['epochs']=epochs
    return params
