import pandas as pd
import math
import numpy as np
df =  pd.read_csv('train.csv').drop(['key'],axis = 1)#.drop(['pickup_datetime'],axis = 1)
accuracy = 0
def convert_time_sec(time):
	time = time.split()[1]
	time = time.split(":")
	return int(time[0])*3600+int(time[1])*60+int(time[2])

df['pickup_datetime'] = df['pickup_datetime'].apply(convert_time_sec)

# print(df_train.mean())
for jj in range(10):
	df_train = df
	df_train = df_train.sample(n = 1000,random_state=np.random.randint(200)).reset_index(drop=True)

	df_test = df_train.sample(n=500,random_state=np.random.randint( 200))
	df_train = df_train.drop(df_test.index)
	df_train = df_train.reset_index(drop=True)
	df_test = df_test.reset_index(drop=True)

	df_test_fare  = df_test['fare_amount']

	df_train_fare = df_train['fare_amount']


	# df_test_fare = df_test['fare_amount']
	train_min = df_train.min()
	train_max = df_train.max()

	classes = []
	step = (train_max-train_min)/5




	df_train = (df_train - train_min)/(train_max - train_min)#scale
	df_test = (df_test - train_min)/(train_max - train_min)
	df_train['fare_amount'] = df_train_fare
	df_test['fare_amount'] = df_test_fare

	for i in range(5):
		classes.append([train_min['fare_amount'],train_min['fare_amount']+step['fare_amount']])

		train_min = train_min+step


	def assign_classes(a,classes = classes):
		for i in range(len(classes)):
			if classes[i][0]<=a and classes[i][1]>=a:
				return i

	df_train['fare_amount'] = df_train['fare_amount'].apply(assign_classes)



	class MLE:
		def __init__ (self, arr,total):
			if arr.shape[0]>1:
				self.cov = np.cov(arr.T)
				self.mean = arr.mean(0)
			self.num = arr.shape[0]
			self.total = total
		def dist(self,vect):
			# print(np.dot(np.dot(np.subtract(vect,self.mean),np.linalg.inv(self.cov)),np.subtract(vect,self.mean).T))
			
			if self.num<2 or np.linalg.det(self.cov)<=0.0 :
				return 0
			return (float(self.num)\
				/float(self.total))*\
			(1.0/math.pow((2*math.pi),1.5))*\
			math.pow((np.linalg.det(self.cov)),-0.5)*\
			math.exp(-0.5*np.dot(np.dot(np.subtract(vect,self.mean),\
				np.linalg.inv(self.cov)),np.subtract(vect,self.mean).T))
	prob_dist = []
	col = ['pickup_datetime','passenger_count']
	for i in range(len(classes)):
		arr = df_train[df_train['fare_amount']==i].as_matrix(columns=col)
		prob_dist.append(MLE(arr,df_train.shape[0]))
	correct = 0
	total = 0
	for i in range(df_test.shape[0]):
		vect =  df_test.iloc[i].as_matrix(columns=col)
		max_prob = 0
		index = 0
		for j in range(len(prob_dist)):

			prob = (prob_dist[j]).dist(vect)
			
			if(prob>max_prob):
				max_prob = prob
				index = j
		actual =  df_test.iloc[i]['fare_amount']
		if actual>=classes[index][0] and actual<=classes[index][1]:
			correct = correct+1
		total = total+1
	accuracy+=(float(correct)/float(total))
print("accuracy: "+str(accuracy/10.0))

