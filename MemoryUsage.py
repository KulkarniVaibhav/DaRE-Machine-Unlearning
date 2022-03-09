import dare
import numpy as np
import time
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score
import sys
import multiprocessing
import os
import joblib
os.environ["OPENBLAS_MAIN_FREE"]="1" 
train_length=10000
test_length=5000
delete_num=5000
dataIdentifier="bank_marketing"

from sklearn.ensemble import RandomForestClassifier
#argorder : dataIdentifier train_length test_length delete_num
if  len(sys.argv)>=2 :
    dataIdentifier=sys.argv[1]
if  len(sys.argv)>=3 :
    train_length=int(sys.argv[2])
if  len(sys.argv)==4 :
    delete_num=int(sys.argv[3])
assert dataIdentifier in ['bank_marketing','adult','AppBehaviour'],"wrong dataset"
assert len(sys.argv) <= 5, "Invalid number of arguments"
assert delete_num<train_length, "Invalid number of elements to delete"

X=np.load("data/"+dataIdentifier+"/train.npy")
print(X.shape)
train_length=min(train_length,X.shape[0])
y=X[:train_length,-1]
X=X[:train_length,:-1]
# X_test=np.load("/home/zubin/dare_rf/data/"+dataIdentifier+"/test.npy")
# y_test=X_test[:test_length,-1]
# X_test=X_test[:test_length,:-1]


def p1(i):
    rf = dare.Forest(n_estimators=400,
                max_features=5, 
                 max_depth=40,
                 k=5,  # no. thresholds to consider per attribute
                 topd=0,  # no. random node layers
                 random_state=1)
    rf.fit(X,y)
    start=time.time()
    rf.delete(np.arange(i))
    end=time.time()
    print(i,end-start," unlearn ")
    # with open("resultsUnlearning"+dataIdentifier+".csv",'a+') as f1:
    #     f1.write(str(i)+","+str(end-start)+"\n")
def p2(i):
    rf2 = dare.Forest(n_estimators=400,
            max_features=5, 
             max_depth=40,
             k=5,  # no. thresholds to consider per attribute
             topd=0,  # no. random node layers
             random_state=1)
    X_temp=X[i:,:]
    y_temp=y[i:]
    start=time.time()
    rf2.fit(X_temp,y_temp)
    end=time.time()
    print(rf2.get_memory_usage())
    # print(i,end-start," retrain ")
    # with open("resultsRetraining"+dataIdentifier+".csv",'a+') as f1:
    #     f1.write(str(i)+","+str(end-start)+"\n")
def p3(i):
    rf2 = dare.Forest(n_estimators=400,
            max_features=5, 
             max_depth=40,
             k=5,  # no. thresholds to consider per attribute
             topd=0,  # no. random node layers
             random_state=1)
    rf2.fit(X[:i,:],y[:i])
    p=rf2.get_memory_usage()
    p=[np.round(x/1024/1024,2) for x in p]
    p_str=[str(x) for x in p]
    clf=RandomForestClassifier(n_estimators=400,max_features=5,max_depth=40)
    clf.fit(X[:i,:],y[:i])
    joblib.dump(clf, "RF_uncompressed.joblib", compress=0) 
    x1=np.round(sum(p),2)
    x2=np.round(os.path.getsize('RF_uncompressed.joblib') / 1024 / 1024, 2)
    return ",".join([str(i),",".join(p_str),str(x1),str(x2),str(np.round(x1/x2,2))])
    

    
    


t=0
iter=5
startIter=50
inc=int((delete_num-startIter)/iter)
# for i in range(startIter,delete_num+1,inc):
    
    
#     t1 = multiprocessing.Process(target=p1, args=(i,))
#     t2 = multiprocessing.Process(target=p2, args=(i,))
    
  
#     t1.start()
#     t2.start()
#     # # t3.start()
#     # # t4.start()
#     # # t5.start()
#     # # t6.start()

  
#     t1.join()
#     t2.join()
#     # # t3.join()
#     # # t4.join()
#     # # t5.join()
#     # # t6.join()
#     # p1(i)
#     # p2(i)
#     t+=1
#     print(t," Done \n")
f=open("memoryUsageStats_"+dataIdentifier+".csv","w")
f.write("DataSetSize,Structure memory,Decision Stats Memory,Leaf Stats memory,Total Memory Usage,ScikitLearn RF MemUsage,Memory Overhead Factor\n")
for i in range(10,1010,100):
    
    f.write(p3(i)+'\n')
    
f.close()

