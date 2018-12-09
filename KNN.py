import pandas as pd
import math
import numpy as np
df_train =  pd.read_csv('train.csv').drop(['key'],axis = 1)#.drop(['pickup_datetime'],axis = 1)


def convert_time_sec(time):
	time = time.split()[1]
	time = time.split(":")
	return int(time[0])*3600+int(time[1])*60+int(time[2])

df_train['pickup_datetime'] = df_train['pickup_datetime'].apply(convert_time_sec)

# print(df_train.mean())
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

df_train = (df_train - train_min)/(train_max - train_min)#scale
df_test = (df_test - train_min)/(train_max - train_min)
df_train['fare_amount'] = df_train_fare
df_test['fare_amount'] = df_test_fare
accuracy = lambda x,y: math.pow((x-y),2.0)
a = 0.0


# print(df_train)
# print(df_test)
for j in range(df_test.shape[0]):
	# print df_test.head(n=2)
	distance = pd.DataFrame(columns=["distance"])
	for i in range (df_train.shape[0]):
		
		distance = distance.append(pd.DataFrame([[math.pow(df_train['pickup_datetime'].iloc[i]-df_test['pickup_datetime'].iloc[j],2.0)+math.pow(df_train['pickup_longitude'].iloc[i]-df_test['pickup_longitude'].iloc[j],2.0)+math.pow(df_train['pickup_latitude'].iloc[i]-df_test['pickup_latitude'].iloc[j],2.0)+math.pow(df_train['dropoff_longitude'].iloc[i]-df_test['dropoff_longitude'].iloc[j],2.0)+math.pow(df_train['dropoff_latitude'].iloc[i]-df_test['dropoff_latitude'].iloc[j],2.0)+math.pow(df_train['passenger_count'].iloc[i]-df_test['passenger_count'].iloc[j],2.0)]],columns= ["distance"],index = [i]))
	
	
	df_train['distance' ] = distance.values

	df_train = df_train.sort_values(["distance"], ascending = True).reset_index(drop=True)
	# print(df_train)

	prediction = df_train['fare_amount'].head(n=5).mean()

	# print(str(prediction)+ " blah "+str(df_test['fare_amount'].iloc[j]))
	
	a= a+accuracy(prediction,df_test['fare_amount'].iloc[j])

print("The predictions are accurate within "+str(math.sqrt(a/float(df_test.shape[0])))+" dollars ")







# print(convert_time_sec('2009-06-15 17:26:21 UTC'))




