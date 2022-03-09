
import os
import sys

import numpy as np
import keras
import dare

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
import math


import matplotlib.pyplot as plt

import tensorflow as tf
import sklearn.metrics as metrics

import multiprocessing
# we'll use keras for neural networks


from keras.utils import np_utils


def is_int(s):
    try:
        z = int(s)
        return z
    except ValueError:
        return None

def random_subdataset(x, y, sz):
    assert x.shape[0] == y.shape[0]
    perm = np.random.permutation(x.shape[0])
    perm = perm[0:sz]

    return x[perm,:].copy(), y[perm,:].copy()
def attack_performance(in_or_out_test, in_or_out_pred):
    cm = metrics.confusion_matrix(in_or_out_test, in_or_out_pred)
    accuracy = np.trace(cm) / np.sum(cm.ravel())
    # tn, fp, fn, tp = cm.ravel()
    # tpr = tp / (tp + fn)
    # fpr = fp / (fp + tn)
    # advantage = tpr - fpr
    return accuracy
def shokri_attack_models(x_aux, y_aux, target_train_size, create_model_fn, train_model_fn, num_shadow=2, attack_model_fn = lambda : LogisticRegression(solver='lbfgs')):
    assert 2*target_train_size < x_aux.shape[0]

    num_classes = y_aux.shape[1]

    class_train_list = [None] * num_classes

    def add_to_list(data):
        for label in range(0, num_classes):
            dv = data[data[:,-2] == label,:]
            col_idx = [i for i in range(0, num_classes)]
            col_idx.append(num_classes+1)
            if class_train_list[label] is None:
                class_train_list[label] = dv[:,col_idx]
            else:
                class_train_list[label] = np.vstack((class_train_list[label], dv[:,col_idx]))

    for i in range(0, num_shadow):
        data_x_combined,data_y_combined=random_subdataset(x_aux,y_aux,2*target_train_size)
        train_data_x,train_data_y=data_x_combined[:target_train_size],data_y_combined[:target_train_size]
        test_data_x,test_data_y=data_x_combined[target_train_size:],data_y_combined[target_train_size:]
        shadow_model=create_model_fn()
        train_model_fn(shadow_model,train_data_x,train_data_y)

        predicted_y_train=shadow_model.predict_proba(train_data_x)
        
        predicted_y_test=shadow_model.predict_proba(test_data_x)
        train_data_y_labels=np.argmax(train_data_y,axis=-1)
        test_data_y_labels=np.argmax(test_data_y,axis=-1)

            
        data=np.column_stack((predicted_y_train,np.argmax(train_data_y,axis=-1),np.ones(target_train_size)))
        data=np.vstack((data,np.column_stack((predicted_y_test,np.argmax(test_data_y,axis=-1),np.zeros(target_train_size)))))

        add_to_list(data)


        

    # now train the models
    attack_models = []

    for label in range(0, num_classes):
        data = class_train_list[label]
        np.random.shuffle(data)
        x_data = data[:,:-1]
        y_data = data[:,-1]

        # train attack model
        am = attack_model_fn().fit(x_data, y_data)
        attack_models.append(am)

    return attack_models

"""
## Perform the Shokri et al. attack
## Inputs:
##  - attack_models: list of attack models, one per class.
##  - x_targets, y_targets: records to attack
##  - query_target_model: function to query the target model [invoke as: query_target_model(x)]

##  Output:
##  - in_or_out_pred: in/out prediction for each target
"""
def train_model(model,x,y):
	model.fit(x,np.argmax(y,axis=-1))
def do_shokri_attack(attack_models, x_targets, y_targets, query_target_model):

    num_classes = y_targets.shape[1]
    assert len(attack_models) == num_classes
    y_targets_labels = np.argmax(y_targets, axis=-1)

    in_or_out_pred = np.zeros((x_targets.shape[0],))

    pv = query_target_model(x_targets)
    assert pv.shape[0] == y_targets_labels.shape[0]

    for i in range(0, pv.shape[0]):
        label = y_targets_labels[i]
        assert 0 <= label < num_classes

        am = attack_models[label]
        in_or_out_pred[i] = am.predict(pv[i,:].reshape(1,-1))

    return in_or_out_pred


def load_preprocess(dataIdentifier,train_in_out_size=2000):
    X=np.load("data/"+dataIdentifier+"/train.npy")
    # let's split the training set further
    aux_idx = train_in_out_size
    y_train=X[:,-1]
    num_classes=2
    y_train=keras.utils.np_utils.to_categorical(y_train, num_classes)
    # X_test=np.load("/home/zubin/dare_rf/data/"+dataIdentifier+"/test.npy")
    # y_test=X_test[:,-1]
    # y_test=keras.utils.np_utils.to_categorical(y_test, num_classes)
    # x_test=X_test[:,:-1]
    x_train=X[:,:-1]

    x_aux = x_train[aux_idx:,:]
    y_aux = y_train[aux_idx:,:]

    x_temp = x_train[:aux_idx,:]
    y_temp = y_train[:aux_idx,:]

    out_idx = int(aux_idx/2.0)
    x_out = x_temp[out_idx:,:]
    y_out = y_temp[out_idx:,:]

    x_train = x_temp[:out_idx,:]
    y_train = y_temp[:out_idx,:]

    return (x_train, y_train), (x_out, y_out),(x_aux, y_aux)
def get_targets(x_out, y_out, sz=2000):

    x_temp = x_out
    y_temp = y_out

    
    outv = np.ones((x_out.shape[0],1))
    in_out_temp = outv

    assert x_temp.shape[0] == y_temp.shape[0]

    if sz > x_temp.shape[0]:
        sz = x_temp.shape[0]

    perm = np.random.permutation(x_temp.shape[0])
    perm = perm[0:sz]
    x_targets = x_temp[perm,:]
    y_targets = y_temp[perm,:]

    in_out_targets = in_out_temp[perm,:]

    return x_targets, y_targets, in_out_targets

def do_loss_attack(x_targets, y_targets, query_target_model, loss_fn, mean_train_loss, std_train_loss, mean_test_loss, std_test_loss):
    pv = query_target_model(x_targets)
    loss_vec = loss_fn(y_targets, pv)

    in_or_out_pred = np.zeros((x_targets.shape[0],))

    for i in range(x_targets.shape[0]) :
        in_prob=np.exp(-(loss_vec[i]-mean_train_loss)**2/(2*std_train_loss**2))/(std_train_loss*np.sqrt(2*np.pi))
        out_prob=np.exp(-(loss_vec[i]-mean_test_loss)**2/(2*std_test_loss**2))/(std_test_loss*np.sqrt(2*np.pi))
        if in_prob>out_prob:
            in_or_out_pred[i]=1
        else:
            in_or_out_pred[i]=0

    return in_or_out_pred


"""
## Perform the loss attack2, assuming the training loss is known
## Inputs:
##  - x_targets, y_targets: records to attack
##  - query_target_model: function to query the target model [invoke as: query_target_model(x)]
##  - loss_fn: function to obtain the loss [invoke as: loss_fn(y, pred_pv)]
##  - mean_train_loss, std_loss: mean and std of the training loss
##  - threshold: decision threshold

##  Output:
##  - in_or_out_pred: in/out prediction for each target
"""
def do_loss_attack2(x_targets, y_targets, query_target_model, loss_fn, mean_train_loss, std_train_loss, threshold=0.9):
    pv = query_target_model(x_targets)
    loss_vec = loss_fn(y_targets, pv)

    in_or_out_pred = np.zeros((x_targets.shape[0],))

    for i in range(x_targets.shape[0]) :
        in_prob=np.exp(-(loss_vec[i]-mean_train_loss)**2/(2*std_train_loss**2))/(std_train_loss*np.sqrt(2*np.pi))
        
        if in_prob>threshold:
            in_or_out_pred[i]=0
        else:
            in_or_out_pred[i]=1
        

    return in_or_out_pred


"""
## Perform the posterior attack
## Inputs:
##  - x_targets, y_targets: records to attack
##  - query_target_model: function to query the target model [invoke as: query_target_model(x)]
##  - threshold: decision threshold

##  Output:
##  - in_or_out_pred: in/out prediction for each target
"""
def do_posterior_attack(x_targets, y_targets, query_target_model, threshold=0.9):
    pv = query_target_model(x_targets)
    confidence_max=np.max(pv,axis=-1)
    in_or_out_pred = np.zeros((x_targets.shape[0],))
    for i in range(x_targets.shape[0]) :
        in_or_out_pred[i]=int(confidence_max[i]>threshold)
    
 

    return in_or_out_pred
def get_dare_model():
	rf=	dare.Forest(n_estimators=200,
                max_features=5, 
                 max_depth=40,
                 k=2,  # no. thresholds to consider per attribute
                 topd=0,  # no. random node layers
                 random_state=1)
	return rf
def compute_loss(y_true, y_pred):
    loss = keras.backend.categorical_crossentropy(tf.convert_to_tensor(y_true,tf.float32), tf.convert_to_tensor(y_pred,tf.float32), from_logits=False)
    return keras.backend.eval(loss)
    #argumentorder : dataIdentifier target_train_size
def main():

    assert len(sys.argv) <= 4, "Incorrect number of arguments"
    dataIdentifier=sys.argv[1]
    train_in_out_size = is_int(sys.argv[2])

    train, out, aux = load_preprocess(dataIdentifier,train_in_out_size=2*train_in_out_size)

    x_train, y_train = train
    target_train_size = x_train.shape[0]
    x_out, y_out = out
 
    x_aux, y_aux = aux
    x_targets, y_targets, in_or_out_targets = get_targets(x_out, y_out)

    target_train_size+=x_targets.shape[0]


    x_train_dare=np.vstack((x_targets,x_train))
    y_train_dare=np.vstack((y_targets,y_train))
    target_model_train_fn=lambda : get_dare_model()
    target_model=target_model_train_fn()
    target_model.fit(x_train_dare,np.argmax(y_train_dare,axis=-1))
    loss_fn = compute_loss
    loss_train_vec = loss_fn(y_train, target_model.predict_proba(x_train))

    target_model.delete(np.arange(len(x_targets)))
    query_target_model = lambda x: target_model.predict_proba(x)
    train_model_fn = lambda model, x, y:train_model(model,x,y)
    create_model_fn=target_model_train_fn
    attack_model_fn = RandomForestClassifier
    attack_models = shokri_attack_models(x_aux, y_aux, target_train_size, create_model_fn, train_model_fn,
                                                    num_shadow=4, attack_model_fn=attack_model_fn)


    in_or_out_pred = do_shokri_attack(attack_models, x_targets, y_targets, query_target_model)
    accuracy =attack_performance(in_or_out_targets, in_or_out_pred)
    print('Shokri attack ({}) accuracy,: {:.1f}%'.format("RandomForestClassifier", 100.0*accuracy))
    f=open("ResultMIA"+dataIdentifier+".csv",'w')
    f.write("Attack Type,Attack Accuracy\n")
    f.write("Shoukri Attack,"+str(100.0*accuracy)+"\n")


    mean_train_loss = np.mean(loss_train_vec)
    std_train_loss = np.std(loss_train_vec)





    threshold_la2=0.7
    in_or_out_pred =do_loss_attack2(x_targets, y_targets, query_target_model, loss_fn, mean_train_loss, std_train_loss,
                                                threshold_la2)

    accuracy = attack_performance(in_or_out_targets, in_or_out_pred)

    print('Loss attack2 accuracy: {:.1f}'.format(100.0*accuracy))
    f.write("Train Loss Threshold attack," + str(100.0 * accuracy) + "\n")
           
       


if __name__ == '__main__':
    main()
