import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dsbase.ConstantsDSBase as constants
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

description='DNNClassification'

# This implementation have some convergency problems for Binary Clasiffication. It might contain a bug. Can you find it?
# Deprecated
class DNNClassificationDSBaseModel:
    def __init__(self, id, X, y, test_perc, parameters, splitter, normalizer):
        self.graph=tf.Graph()
        with self.graph.as_default():
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
            
            self.input_size=X.shape[1]
            self.output_size=1 # TODO: multivariable can not be done
            self.layers=self.parameters['layers']
            self.alpha=self.parameters['alpha']
            self.beta1=self.parameters['beta1']            
            self.beta2=self.parameters['beta2']
            self.epsylon=self.parameters['epsylon']
            
            # Placeholders
            self.X = tf.placeholder(dtype=tf.float32, shape=[None,self.input_size])
            self.y = tf.placeholder(dtype=tf.float32, shape=[None,1])   # TODO: multivariable can not be done
            
            # Variables
            self.ws=[]
            self.bs=[]
            prev_l=self.input_size
            for l in self.layers:
                #w=tf.Variable(tf.truncated_normal(shape=[prev_l,l],stddev=0.1))
                #b=tf.Variable(tf.truncated_normal(shape=[l],stddev=0.1))
                w=tf.get_variable("w" + str(l),shape=[prev_l,l],initializer=tf.contrib.layers.xavier_initializer())
                b=tf.get_variable("b" + str(l), shape=[l],initializer=tf.contrib.layers.xavier_initializer())
                self.ws.append(w)
                self.bs.append(b)
                prev_l=l
            #w=tf.Variable(tf.truncated_normal(shape=[prev_l,1],stddev=0.1))
            #b=tf.Variable(tf.truncated_normal(shape=[1],stddev=0.1))
            w=tf.get_variable("w",shape=[prev_l,1],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable("b",shape=[1],initializer=tf.contrib.layers.xavier_initializer())            
            self.ws.append(w)
            self.bs.append(b)

            # Operations
            self.out_data=self.__forward()
            
            # Loss Function 
            if (y.max() == 1):
                self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=self.out_data))
            else:
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.out_data))
            
            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(self.alpha, self.beta1, self.beta2, self.epsylon)
            #self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
            self.model = self.optimizer.minimize(self.cost)

            # Persistance manager
            self.saver = tf.train.Saver()

            self.session = tf.Session()
        
    def train(self):
        with self.graph.as_default():
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
            plt.title('cost DNN Model ' + str(self.id) + ". alpha=" + str(self.alpha) + ", beta1: " + str(self.beta1) + ", beta2: " + str(self.beta2) + ",epsylon: " + str(self.epsylon))
            #plot.show()
            
        
    def predict(self, test_data):
        with self.graph.as_default():
            print("predicting model " + str(self.id) + ". " + description);
            #tf.reset_default_graph()
            test_data_scaled=self.scalerX.transform(test_data)
            output=self.session.run(self.__forward(),feed_dict={self.X:test_data_scaled})
            return self.scalery.inverse_transform(output)
        
    def getTrainScore(self):
        with self.graph.as_default():
            output=self.__score(self.X_train,self.y_train)
            return output
    
    def getTestScore(self):
        with self.graph.as_default():
            output=self.__score(self.X_test,self.y_test)
            return output
        
    def save(self, folder_path=constants.PERSISTANCE_FOLDER):
        with self.graph.as_default():
            file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT
            print("saving model: " + file_path)
            self.saver.save(self.session, file_path)
            scaler_file_path_root=folder_path + constants.SEP + description + "_" + str(self.id)
            joblib.dump(self.scalerX, scaler_file_path_root + "_scalerX" + constants.EXT)
            joblib.dump(self.scalery, scaler_file_path_root + "_scalery" + constants.EXT)
    
    def load(self, folder_path=constants.PERSISTANCE_FOLDER):
        with self.graph.as_default():
            file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT
            print("loading model: " + file_path)
            self.saver = tf.train.import_meta_graph(file_path + '.meta')
            self.saver.restore(self.session,tf.train.latest_checkpoint(folder_path + constants.SEP))
            #self.saver.restore(self.session, file_path)
            #print('ws_0: ' + str(self.ws[0].eval(self.session)))
            scaler_file_path_root=folder_path + constants.SEP + description + "_" + str(self.id)
            self.scalerX=joblib.load(scaler_file_path_root + "_scalerX" + constants.EXT)
            self.scalery=joblib.load(scaler_file_path_root + "_scalery" + constants.EXT)

    def close(self):
        self.session.close()
        self.graph.finalize()
    
    def __forward(self):
        with self.graph.as_default():
            in_data=self.X
            for i,l in enumerate(self.layers):
                #print(str(in_data.shape) + ", " + str(self.ws[i].shape) + ", " + str(self.bs[i].shape))
                out_data=tf.nn.sigmoid(tf.matmul(in_data,self.ws[i])+self.bs[i])
                in_data=out_data
            out_data=tf.matmul(in_data,self.ws[-1])+self.bs[-1]
            return out_data

    
    def __score(self,X, y):
        with self.graph.as_default():
            yp_prev_1 = self.__forward()
            yp_prev_2 = tf.nn.sigmoid(yp_prev_1)
            yp = tf.round(yp_prev_2)
            #C = tf.reduce_sum(tf.abs(tf.subtract(tf.cast(y, dtype=tf.float32),tf.cast(yp, dtype=tf.float32))))
            #P = tf.subtract(tf.constant(1,dtype=tf.float32),tf.divide(C,len(y)))
            #P = tf.divide(C,len(y))
            #init_l = tf.local_variables_initializer()
            #self.session.run(init_l)
            preds=self.session.run(yp,feed_dict={self.X:X})
            output=np.sum(np.abs(y-preds))/len(y)
            return output

# Params converter function. Reference for every model
def DNNClassificationDSBaseParamsToMap(layers, alpha=1e-2, beta1=0.9, beta2=0.999, epsylon=1e-8,batch_size=128, epochs=10):
    params={}
    params['layers']=layers
    params['alpha']=alpha
    params['beta1']=beta1
    params['beta2']=beta2
    params['epsylon']=epsylon       
    params['batch_size']=batch_size
    params['epochs']=epochs
    return params
