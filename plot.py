
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
def main():
	dataIdentifier=sys.argv[1]
	assert dataIdentifier!="AppBehaviour", "Error"
	X=np.load("/home/zubin/dare_rf/data/"+dataIdentifier+"/train.npy")
	datasetSize=X.shape[0]
	df1 = pd.read_csv("resultsUnlearning"+dataIdentifier+".csv",header=None,names=["percentage of data deleted","Computation Time for Operation"])
	df2 = pd.read_csv("resultsRetraining"+dataIdentifier+".csv",header=None,names=["percentage of data deleted","Computation Time for Operation"])


	plt.figure(figsize=(16,10))
	x1=df1["percentage of data deleted"].to_numpy()*100.0/datasetSize
	y1=df1["Computation Time for Operation"].to_numpy()
	x2=df2["percentage of data deleted"].to_numpy()*100.0/datasetSize
	y2=df2["Computation Time for Operation"].to_numpy()
	plt.plot(x1,y1,color='r',label = 'Unlearning Computation')
	plt.plot(x2,y2,color='b',label = 'Retraining Computation')
	plt.scatter(x1,y1,color='r', s=10)
	plt.scatter(x2,y2, color='b',s=10)

	plt.title("Plot of Time taken vs  percentage of elements \n deleted through either Retraining or Unlearning for"+dataIdentifier+" datset",fontsize=26)
	plt.xlabel("Percentage of elements deleted (in %) ",fontsize=22)
	plt.ylabel("Computation Time for Operation (in seconds)",fontsize=22)
	plt.locator_params(axis="x",nbins=20)
	plt.yticks(np.arange(0,20,1))
	plt.legend()
	plt.savefig("PlotUnlearningVsRetraining"+dataIdentifier+".png")

if __name__=="__main__":
	main()
