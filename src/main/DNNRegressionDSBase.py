import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ConstantsDSBase as constants
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

description='DNNRegression'

class DNNRegressionDSBaseModel:
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
        else:
            print("initiating empty model " + str(self.id) + ". " + description);
            tf.reset_default_graph()
        
        self.input_size=X.shape[1]
        self.output_size=1 # TODO: multivariable can not be done
        self.layers=self.parameters['layers']
        self.alpha=self.parameters['alpha']
        
        # Placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=[None,self.input_size])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None,1])   # TODO: multivariable can not be done
        
        # Variables
        self.ws=[]
        self.bs=[]
        prev_l=self.input_size
        for l in self.layers:
            w=tf.Variable(tf.truncated_normal(shape=[prev_l,l],stddev=0.1))
            b=tf.Variable(tf.truncated_normal(shape=[l],stddev=0.1))
            self.ws.append(w)
            self.bs.append(b)
            prev_l=l
        w=tf.Variable(tf.truncated_normal(shape=[prev_l,1],stddev=0.1))
        b=tf.Variable(tf.truncated_normal(shape=[1],stddev=0.1))
        self.ws.append(w)
        self.bs.append(b)

        # Operations
        self.out_data=self.__forward()
        
        # Loss Function 
        self.cost = tf.reduce_mean(tf.square(self.out_data-self.y))
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.alpha)
        self.model = self.optimizer.minimize(self.cost)

        # Persistance manager
        self.saver = tf.train.Saver()
        
    def train(self):
        print("training model " + str(self.id) + ". " + description);
        #tf.reset_default_graph()
        init = tf.global_variables_initializer()

        self.session.run(init)

        batch_size=self.parameters['batch_size']
        epochs=self.parameters['epochs']

        batches=int(self.X_train.shape[0]/batch_size)

        cost = []
        for epoch in range(epochs):
            for batch in range(batches):
                XBatch = self.X_train[batch*batch_size:(batch+1)*batch_size,:]
                yBatch = self.y_train[batch*batch_size:(batch+1)*batch_size,:]   # TODO: Multivariable can not be done
                self.session.run(self.model,feed_dict={self.X:XBatch,self.y:yBatch})
                c, self.p = self.session.run((self.cost,self.__forward()),feed_dict={self.X:XBatch,self.y:yBatch})
                if batch % 10 == 0:
                    print("epoch " + str(epoch) + ". batch / n_batches:", batch, "/", batches, "cost:", c)
                cost.append(c)
        #print('ws_0: ' + str(self.ws[0].eval(self.session)))
        plt.plot(cost)
        plt.title('cost DNN Model ' + str(self.id) + ". alpha=" + str(self.alpha))
        #plot.show()
            
        
    def predict(self, test_data):
        print("predicting model " + str(self.id) + ". " + description);
        #tf.reset_default_graph()
        test_data_scaled=self.scalerX.transform(test_data)
        output=self.session.run(self.__forward(),feed_dict={self.X:test_data_scaled})
        return self.scalery.inverse_transform(output)
        
    def getTrainScore(self):
        output=self.__score_R2(self.X_train,self.y_train)
        return output
    
    def getTestScore(self):
        output=self.__score_R2(self.X_test,self.y_test)
        return output
        
    def save(self, folder_path=constants.PERSISTANCE_FOLDER):
        file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT
        print("saving model: " + file_path)
        self.saver.save(self.session, file_path)
        scaler_file_path_root=folder_path + constants.SEP + description + "_" + str(self.id)
        joblib.dump(self.scalerX, scaler_file_path_root + "_scalerX" + constants.EXT)
        joblib.dump(self.scalery, scaler_file_path_root + "_scalery" + constants.EXT)
    
    def load(self, folder_path=constants.PERSISTANCE_FOLDER):
        file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT
        print("loading model: " + file_path)
        self.saver = tf.train.import_meta_graph(file_path + '.meta')
        self.saver.restore(self.session,tf.train.latest_checkpoint(folder_path + constants.SEP))
        #self.saver.restore(self.session, file_path)
        #print('ws_0: ' + str(self.ws[0].eval(self.session)))
        scaler_file_path_root=folder_path + constants.SEP + description + "_" + str(self.id)
        self.scalerX=joblib.load(scaler_file_path_root + "_scalerX" + constants.EXT)
        self.scalery=joblib.load(scaler_file_path_root + "_scalery" + constants.EXT)
    
    def setSession(self,sess):
        self.session=sess

    def startSession(self):
        self.session = tf.Session()

    def closeSession(self):
        self.session.close()
    
    def __forward(self):
        in_data=self.X
        for i,l in enumerate(self.layers):
            #print(str(in_data.shape) + ", " + str(self.ws[i].shape) + ", " + str(self.bs[i].shape))
            out_data=tf.nn.sigmoid(tf.matmul(in_data,self.ws[i])+self.bs[i])
            in_data=out_data
        out_data=tf.matmul(in_data,self.ws[-1])+self.bs[-1]
        return out_data
    
    def __score_R2(self,X, y):
        yp = self.__forward()
        u = tf.cast(tf.reduce_sum(tf.square(y - yp)), dtype=tf.float32)
        v = tf.cast(tf.reduce_sum(tf.square(y - tf.reduce_mean(y))), dtype=tf.float32)
        R = tf.subtract(tf.constant(1,dtype=tf.float32),tf.divide(u,v))
        output=self.session.run(R,feed_dict={self.X:X})
        return output

# Params converter function. Reference for every model
def DNNRegressionDSBaseParamsToMap(layers, alpha=1e-2, batch_size=128, epochs=10):
    params={}
    params['layers']=layers
    params['alpha']=alpha
    params['batch_size']=batch_size
    params['epochs']=epochs
    return params
