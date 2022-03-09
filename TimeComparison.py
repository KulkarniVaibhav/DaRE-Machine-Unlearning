import dare
import numpy as np
import time
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score
import sys
import joblib
train_length=10000
test_length=5000
delete_num=5000
dataIdentifier="bank_marketing"

#argorder : dataIdentifier train_length test_length delete_num
if  len(sys.argv)>=2 :
    dataIdentifier=sys.argv[1]
if  len(sys.argv)>=3 :
    train_length=int(sys.argv[2])
if  len(sys.argv)>=4 :
    test_length=int(sys.argv[3])
if  len(sys.argv)==5 :
    delete_num=int(sys.argv[4])
assert dataIdentifier in ['bank_marketing','adult','AppBehaviour'],"wrong dataset"
assert len(sys.argv) <= 5, "Invalid number of arguments"
assert delete_num<train_length, "Invalid number of elements to delete"

X=np.load("data/"+dataIdentifier+"/train.npy")
print(X.shape)
y=X[:train_length,-1]
X=X[:train_length,:-1]
X_test=np.load("data/"+dataIdentifier+"/test.npy")
y_test=X_test[:test_length,-1]
X_test=X_test[:test_length,:-1]


precision_scores=[]
recall_scores=[]
accuracy_scores=[]
times=[]
# train a DaRE RF model
rf = dare.Forest(n_estimators=400,
                max_features=5, 
                 max_depth=40,
                 k=2,  # no. thresholds to consider per attribute
                 topd=0,  # no. random node layers
                 random_state=1)
# rf = dare.Forest(n_estimators=100,
#                 max_features=10, 
#                  max_depth=3,
#                  k=5,  # no. thresholds to consider per attribute
#                  topd=0,  # no. random node layers
#                  random_state=1)
rf.fit(X, y)
# joblib.dump(rf,open('rfModel1.sav','wb'))
f=open("results.txt","a")
# print(rf.sim_delete(train_length))





y_pred=np.argmax(rf.predict_proba(X),axis=-1)
conf_matrix = confusion_matrix(y_true=y, y_pred=y_pred)
precision_scores.append(precision_score(y_true=y, y_pred=y_pred))
recall_scores.append(recall_score(y_true=y, y_pred=y_pred))
accuracy_scores.append(accuracy_score(y_true=y, y_pred=y_pred))
f.write("Training data confusion matrix before deletion for "+dataIdentifier+" Datset with the following attributes : "+" ,training set data size :"+str(train_length)+" ,test set data size :"+str(test_length)+" ,number of records to delete: " +str(delete_num)+"\n" +str(conf_matrix)+"\n"
    +",precision_score :"+str(precision_scores[0])+",recall_score:"+str(recall_scores[0])+",accuracy_score : "+str(accuracy_scores[0])+"\n")
# # prediction before deletion => [0.5, 0.5]
y_pred=np.argmax(rf.predict_proba(X_test),axis=-1)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
precision_scores.append(precision_score(y_true=y_test, y_pred=y_pred))
recall_scores.append(recall_score(y_true=y_test, y_pred=y_pred))
accuracy_scores.append(accuracy_score(y_true=y_test, y_pred=y_pred))
f.write("Testing data confusion matrix before deletion for "+dataIdentifier+" Datset with the following attributes :"+" ,training set data size :"+str(train_length)+" ,test set data size :"+str(test_length)+" ,number of records to delete: " +str(delete_num)+"\n"+str(conf_matrix)+"\n"
    +",precision_score :"+str(precision_scores[1])+",recall_score:"+str(recall_scores[1])+",accuracy_score : "+str(accuracy_scores[1])+"\n")

# # delete training example at index 3 ([1, 0], 0)
start=time.time()
# for i in range(4):
#     rf.delete(np.arange(i*delete_num/4,(i+1)*delete_num/4))
rf.delete(np.arange(delete_num))
end=time.time()
times.append(end-start)
with open("results2.csv",'a') as f1:
    f1.write(str(delete_num)+","+str(times[len(times)-1])+'\n')
# # prediction after deletion => [0.0, 1.0]
# rf.predict_proba(X_test)
y_pred=np.argmax(rf.predict_proba(X),axis=-1)
conf_matrix = confusion_matrix(y_true=y, y_pred=y_pred)
precision_scores.append(precision_score(y_true=y, y_pred=y_pred))
recall_scores.append(recall_score(y_true=y, y_pred=y_pred, ))
accuracy_scores.append(accuracy_score(y_true=y, y_pred=y_pred))
f.write("Training data confusion matrix after deletion for "+dataIdentifier+" Datset with the following attributes :"+" ,training set data size :"+str(train_length)+" ,test set data size :"+str(test_length)+" ,number of records to delete: " +str(delete_num)+"\n" +str(conf_matrix)+"\n"
    +",precision_score :"+str(precision_scores[2])+",recall_score:"+str(recall_scores[2])+",accuracy_score : "+str(accuracy_scores[2])+"\n")
y_pred=np.argmax(rf.predict_proba(X_test),axis=-1)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
precision_scores.append(precision_score(y_true=y_test, y_pred=y_pred))
recall_scores.append(recall_score(y_true=y_test, y_pred=y_pred))
accuracy_scores.append(accuracy_score(y_true=y_test, y_pred=y_pred))
f.write("Testing data confusion matrix after deletion for "+dataIdentifier+" Datset with the following attributes :"+" ,training set data size :"+str(train_length)+" ,test set data size :"+str(test_length)+" ,number of records to delete: " +str(delete_num)+"\n" +str(conf_matrix)+"\n"
    +",precision_score :"+str(precision_scores[3])+",recall_score:"+str(recall_scores[3])+",accuracy_score : "+str(accuracy_scores[3])+"\n")
f.write("\n generlization accuracy decrease :" +str(100*(accuracy_scores[1]-accuracy_scores[3])/accuracy_scores[1])+"%")
f.write("\n")
# print(rf.get_node_statistics())
# print(rf.get_memory_usage())
rf2= dare.Forest(n_estimators=400,
                max_features=5, 
                 max_depth=40,
                 k=2,  # no. thresholds to consider per attribute
                 topd=0,  # no. random node layers
                 random_state=1)
start=time.time()
rf2.fit(X[delete_num:,:],y[delete_num:])
end=time.time()
times.append(end-start)
y_pred=np.argmax(rf2.predict_proba(X),axis=-1)
conf_matrix = confusion_matrix(y_true=y, y_pred=y_pred)
precision_scores.append(precision_score(y_true=y, y_pred=y_pred))
recall_scores.append(recall_score(y_true=y, y_pred=y_pred))
accuracy_scores.append(accuracy_score(y_true=y, y_pred=y_pred))
f.write("Train data stats for deletion on retrained "+dataIdentifier+" Datset with the following attributes :"+" ,training set data size :"+str(train_length)+" ,test set data size :"+str(test_length)+" ,number of records to delete: " +str(delete_num)+"\n" +str(conf_matrix)+"\n"
    +",precision_score :"+str(precision_scores[4])+",recall_score:"+str(recall_scores[4])+",accuracy_score : "+str(accuracy_scores[4])+"\n")
y_pred=np.argmax(rf2.predict_proba(X_test),axis=-1)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
precision_scores.append(precision_score(y_true=y_test, y_pred=y_pred ))
recall_scores.append(recall_score(y_true=y_test, y_pred=y_pred))
accuracy_scores.append(accuracy_score(y_true=y_test, y_pred=y_pred))
f.write("Test data stats for deletion on retrained "+dataIdentifier+" Datset with the following attributes :"+" ,training set data size :"+str(train_length)+" ,test set data size :"+str(test_length)+" ,number of records to delete: " +str(delete_num)+"\n" +str(conf_matrix)+"\n"
    +",precision_score :"+str(precision_scores[5])+",recall_score:"+str(recall_scores[5])+",accuracy_score : "+str(accuracy_scores[5])+"\n")
f.write("Times for Unlearn is"+str(1000*times[0])+"ms vs retrain :"+str(1000*times[1])+"ms")
f.write("\n\n\n\n")
f.close()

