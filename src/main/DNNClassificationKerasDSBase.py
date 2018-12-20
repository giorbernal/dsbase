import numpy as np
import matplotlib.pyplot as plt 
import ConstantsDSBase as constants
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense

description='DNNClassificationKeras'

class DNNClassificationKerasDSBaseModel:
    def __init__(self, id, X, y, test_perc, parameters, splitter, normalizer):
        self.id=id
        self.parameters=parameters
        if (test_perc is not None):
            print("initiating model " + str(self.id) + ". " + description);
            # Normalizing
            self.scalerX=MinMaxScaler()
            X_s = self.scalerX.fit_transform(X)
            self.scalery=MinMaxScaler()
            y_s = self.scalery.fit_transform(y.reshape(-1, 1))
           
            X_train, X_test, y_train, y_test = splitter(X_s, y_s, test_size=test_perc, random_state=42)
            self.X_train=X_train
            self.X_test=X_test
            self.y_train=y_train
            self.y_test=y_test

            self.input_size=X.shape[1]
            if (max(y)==1):
                self.output_size=1
            else:
                self.output_size=max(y)

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
        test_data_norm = self.scalerX.transform(test_data)
        results = self.model.predict_classes(test_data_norm)
        return self.scalery.inverse_transform(results)

    def getTrainScore(self):
        return self.model.evaluate(self.X_train, self.y_train, batch_size=self.batch_size)[1]
    
    def getTestScore(self):
        return self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)[1]
        
    def save(self, folder_path=constants.PERSISTANCE_FOLDER):
        file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT  # .h5 file
        self.model.save(file_path)
        scaler_file_path_root=folder_path + constants.SEP + description + "_" + str(self.id)
        joblib.dump(self.scalerX, scaler_file_path_root + "_scalerX" + constants.EXT)
        joblib.dump(self.scalery, scaler_file_path_root + "_scalery" + constants.EXT)

    
    def load(self, folder_path=constants.PERSISTANCE_FOLDER):
        file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT  # .h5 file
        self.model = load_model(file_path)
        scaler_file_path_root=folder_path + constants.SEP + description + "_" + str(self.id)
        self.scalerX=joblib.load(scaler_file_path_root + "_scalerX" + constants.EXT)
        self.scalery=joblib.load(scaler_file_path_root + "_scalery" + constants.EXT)

    def close(self):
        pass


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
